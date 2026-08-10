"""
Headless TurtleBot3 + Nav2 rollout, with an RGB camera bridged to ROS.

Phase A of the record-first/infer-later split. This launch never starts a GUI:
the Gazebo *client* is what actually kills this laptop (software-rendered OpenGL
on a 15 W chip with no /dev/dri), while `gz sim -s` plus offscreen sensor
rendering is affordable.

Two things it adds on top of the stock nav2 simulation:

  1. the robot description is this repo's vendored `waffle_rgbd.sdf.xacro`,
     which has an RGB camera the stock waffle does not have;
  2. an extra ros_gz_bridge for that camera, since the stock bridge config
     forwards no image topic.

    ros2 launch rollout.launch.py               # sim + nav2, headless
    ros2 launch rollout.launch.py nav:=false    # sim only (teleop/manual drive)
"""

import os
import subprocess
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (AndSubstitution, Command, FindExecutable,
                                  LaunchConfiguration, NotSubstitution)
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
RGBD_SDF = str(HERE.parent / "models" / "waffle_rgbd.sdf.xacro")


def _launch_gz_server(context, *_args, **_kwargs):
    """Expand the world xacro to a real file, then start the server on it."""
    world_path = LaunchConfiguration("world").perform(context)

    if world_path.endswith((".xacro", ".sdf.xacro")):
        expanded = subprocess.run(
            ["xacro", world_path], capture_output=True, text=True, check=True
        ).stdout
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", prefix="rollout_world_", delete=False
        )
        tmp.write(expanded)
        tmp.close()
        world_path = tmp.name

    return [
        ExecuteProcess(
            cmd=["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "2", world_path],
            output="screen",
            shell=False,
        )
    ]


def generate_launch_description():
    sim_dir = get_package_share_directory("nav2_minimal_tb3_sim")
    nav2_dir = get_package_share_directory("nav2_bringup")

    world = LaunchConfiguration("world")
    nav = LaunchConfiguration("nav")
    use_nav = IfCondition(nav)

    args = [
        DeclareLaunchArgument(
            "world",
            default_value=os.path.join(sim_dir, "worlds", "tb3_sandbox.sdf.xacro"),
            description="world file (xacro or sdf)",
        ),
        DeclareLaunchArgument("nav", default_value="true", description="also bring up Nav2"),
        # Build the map instead of localising in a prebuilt one. The thing that
        # must never happen is slam_toolbox and AMCL running together: both
        # publish map->odom, and two publishers on one transform do not error,
        # they tear the map silently. So the combinations are:
        #
        #   nav=false slam=false   bare sim (p1 drives /cmd_vel open-loop)
        #   nav=false slam=true    sim + standalone slam_toolbox (p2 mapping run)
        #   nav=true  slam=false   Nav2 with AMCL against a prebuilt map
        #   nav=true  slam=true    Nav2 in SLAM mode; nav2_bringup runs
        #                          slam_toolbox itself and skips AMCL, so the
        #                          standalone node below must stand down
        DeclareLaunchArgument("slam", default_value="false",
                              description="build a map with slam_toolbox instead of "
                                          "localising with AMCL"),
        DeclareLaunchArgument("x_pose", default_value="-2.00"),
        DeclareLaunchArgument("y_pose", default_value="-0.50"),
        DeclareLaunchArgument("yaw", default_value="0.00"),
        DeclareLaunchArgument(
            "map",
            default_value=os.path.join(nav2_dir, "maps", "tb3_sandbox.yaml"),
            description="map for AMCL localisation",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(nav2_dir, "params", "nav2_params.yaml"),
        ),
    ]

    # Software rendering is the only option here; be explicit rather than
    # relying on whatever the shell happened to export.
    env = [
        SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
        AppendEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH", os.path.join(sim_dir, "models")
        ),
        AppendEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH", str(Path(sim_dir).parent.resolve())
        ),
    ]

    # -s = server only, no GUI. --headless-rendering keeps sensor rendering
    # offscreen via EGL, which is what makes the camera work without /dev/dri.
    #
    # `gz sim` takes a world *path*, not world XML — handing it the expanded
    # xacro output makes it treat the whole document as a filename, fail to find
    # it, and then try to download it from Fuel. So expand to a real file first.
    gz_server = OpaqueFunction(function=_launch_gz_server)

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_dir, "launch", "spawn_tb3.launch.py")
        ),
        launch_arguments={
            "robot_sdf": RGBD_SDF,
            "x_pose": LaunchConfiguration("x_pose"),
            "y_pose": LaunchConfiguration("y_pose"),
            "yaw": LaunchConfiguration("yaw"),
        }.items(),
    )

    # The stock bridge config has no image entry, so bridge the camera here.
    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="rgb_camera_bridge",
        arguments=["/rgb_camera@sensor_msgs/msg/Image[gz.msgs.Image"],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # Ground-truth pose straight from the simulator.
    #
    # /odom cannot be ground truth here: the DiffDrive plugin integrates wheel
    # rotation, so a robot pinned against a wall with its wheels spinning
    # reports metres of travel it never made. Measured on the `stuck_wall` run —
    # 5.66 m of "movement" while physically stationary. Labelling failures from
    # odometry would therefore mislabel the exact scenario this project is about.
    #
    # The source is the PosePublisher plugin added to waffle_rgbd.sdf.xacro,
    # which publishes the model's true pose on /model/<name>/pose. The world's
    # dynamic_pose/info topic was tried first and does not reliably deliver here.
    gt_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gt_pose_bridge",
        arguments=[
            "/model/turtlebot3_waffle/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"
        ],
        remappings=[("/model/turtlebot3_waffle/pose", "/gt_pose")],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "robot_description": Command(
                    [
                        FindExecutable(name="xacro"), " ",
                        os.path.join(sim_dir, "urdf", "turtlebot3_waffle.urdf"),
                    ]
                ),
            }
        ],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_dir, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "map": LaunchConfiguration("map"),
            "use_sim_time": "True",
            "params_file": LaunchConfiguration("params_file"),
            "autostart": "True",
            # False, deliberately. With composition Nav2 starts one container
            # and loads AMCL, map_server and the planners into it over a
            # service. On this box that load never completed: the container came
            # up, no nodes appeared inside it, and the run hung waiting for
            # amcl/get_state forever. Separate processes cost more RAM and start
            # slower, but they actually start.
            "use_composition": "False",
            "use_respawn": "False",
            # True makes nav2_bringup run slam_toolbox in place of AMCL.
            "slam": LaunchConfiguration("slam"),
        }.items(),
        condition=use_nav,
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch", "online_async_launch.py",
            )
        ),
        launch_arguments={"use_sim_time": "True"}.items(),
        # Only when Nav2 is absent. With nav:=true, nav2_bringup owns the
        # slam_toolbox instance and a second one here would fight it for
        # map->odom.
        # AndSubstitution/NotSubstitution, not PythonExpression: the latter
        # eval()s its text, and launch substitutes the lowercase strings "true"
        # and "false", which are not Python literals. It fails at launch time
        # with "name 'false' is not defined".
        condition=IfCondition(AndSubstitution(
            LaunchConfiguration("slam"),
            NotSubstitution(LaunchConfiguration("nav")),
        )),
    )

    return LaunchDescription(
        args
        + env
        + [gz_server, spawn, camera_bridge, gt_pose_bridge, robot_state_publisher,
           nav2, slam]
    )

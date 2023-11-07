add_library(${PROJECT_NAME}.tools
        pointcloud_convert/velodyne_convertor.cc
        pointcloud_convert/packets_parser.cc
        ui/pangolin_window.cc
        ui/pangolin_window_impl.cc
        ui/ui_car.cc
        ui/ui_trajectory.cc
        ui/ui_cloud.cc
        )

target_link_libraries(${PROJECT_NAME}.tools
        ${third_party_libs}

        ${PROJECT_SOURCE_DIR}/build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbb.so
        ${PROJECT_SOURCE_DIR}/build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc.so
        ${PROJECT_SOURCE_DIR}/build/tbb_cmake_build/tbb_cmake_build_subdir_release/libtbbmalloc_proxy.so
        )

add_executable(ui_test ui/ui_test.cc)
target_link_libraries(ui_test
        ${PROJECT_NAME}.tools
        ${third_party_libs}
        )

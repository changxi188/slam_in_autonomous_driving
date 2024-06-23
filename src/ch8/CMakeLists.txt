message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

add_library(${PROJECT_NAME}.ch8
        lio-iekf/lio_iekf.cc
        lio-preinteg/lio_preinteg.cc
        ikd-Tree/ikd_Tree.cpp
        )

target_link_libraries(${PROJECT_NAME}.ch8
        ${PROJECT_NAME}.ch7
        ${PROJECT_NAME}.ch3
        ${PROJECT_NAME}.common
        ${third_party_libs}
        )

add_executable(test_lio_iekf test/test_lio_iekf.cc)
target_link_libraries(test_lio_iekf
        ${PROJECT_NAME}.ch8
        ${PROJECT_NAME}.ch7
        ${PROJECT_NAME}.ch4
        )

    #add_executable(test_lio_preinteg test/test_lio_preinteg.cc)
    #target_link_libraries(test_lio_preinteg
    #        ${PROJECT_NAME}.ch8
    #        ${PROJECT_NAME}.ch7
    #        ${PROJECT_NAME}.ch4
    #        )

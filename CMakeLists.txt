# CMakeList.txt: IndustrialVision 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required(VERSION 3.21)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(IndustrialVision LANGUAGES CXX)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# 查找外部依赖
find_package(OpenCV REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# 建立视觉代码库
add_library(vision_core
    src/MaskUtils.cpp
    include/industrial_vision/MaskUtils.h
    src/ImageSequenceSource.cpp
    include/industrial_vision/ImageSequenceSource.h
    src/AnnotationPreview.cpp
    include/industrial_vision/AnnotationPreview.h
    src/TraditionalInspector.cpp
    include/industrial_vision/TraditionalInspector.h
  )
# 取了一个名字
add_library(IndustrialVision::VisionCore ALIAS vision_core)

# 最低 C++ 标准要求
target_compile_features(vision_core PUBLIC cxx_std_17)

# 如何找到头文件和库链接
target_include_directories(vision_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS})
# 生成可执行文件
add_executable(IndustrialVision src/IndustrialVision.cpp)
target_link_libraries(IndustrialVision PRIVATE IndustrialVision::VisionCore)

# 生成 Qt 可执行文件
add_executable(IndustrialVisionQt
    src/QtMain.cpp
    src/ui/MainWindow.cpp
    src/ui/MainWindow.h
    src/ui/BatchWorker.cpp
    src/ui/BatchWorker.h
    src/ui/LiveWorker.cpp
    src/ui/LiveWorker.h
  )

set_target_properties(IndustrialVisionQt PROPERTIES AUTOMOC ON)

# 设置 Qt 可执行文件的 C++ 标准以及头文件和库链接
target_link_libraries(IndustrialVisionQt
    PRIVATE IndustrialVision::VisionCore Qt6::Widgets)

# 查看是否设置了 ONNXRUNTIME_ROOT 环境变量
if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT in CMakeUserPresets.json")
endif()
add_library(vision_inference
    src/YoloInspector.cpp
    include/industrial_vision/YoloInspector.h
  )
add_executable(LiveProbe src/LiveProbe.cpp)
target_link_libraries(LiveProbe PRIVATE vision_inference)


target_compile_features(vision_inference PUBLIC cxx_std_17)

target_include_directories(vision_inference PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(vision_inference
    PUBLIC IndustrialVision::VisionCore
    PRIVATE "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)

target_link_libraries(IndustrialVisionQt PRIVATE vision_inference)

# 生成 ModelProbe 可执行文件
add_executable(ModelProbe src/ModelProbe.cpp)
# 设置 C++ 标准以及头文件和库链接
target_compile_features(ModelProbe PRIVATE cxx_std_17)

target_include_directories(ModelProbe PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ModelProbe PRIVATE
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    ${OpenCV_LIBS}
)

# 复制 ONNX Runtime DLL 到 ModelProbe 可执行文件所在目录
add_custom_command(TARGET ModelProbe POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:ModelProbe>"
    VERBATIM

)

if(WIN32)
    if(NOT DEFINED OpenCV_RUNTIME_DIR)
        message(FATAL_ERROR "Set OpenCV_RUNTIME_DIR in your local CMake user preset.")
    endif()

    file(GLOB OpenCV_WORLD_DLLS "${OpenCV_RUNTIME_DIR}/opencv_world*.dll")
    if(NOT OpenCV_WORLD_DLLS)
        message(FATAL_ERROR "No OpenCV world runtime DLLs found in: ${OpenCV_RUNTIME_DIR}")
    endif()

    add_custom_command(TARGET LiveProbe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        ${OpenCV_WORLD_DLLS}
        "$<TARGET_FILE_DIR:LiveProbe>"
    COMMAND_EXPAND_LISTS
    VERBATIM
)

    add_custom_command(TARGET IndustrialVisionQt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        ${OpenCV_WORLD_DLLS}
        "$<TARGET_FILE_DIR:IndustrialVisionQt>"
    COMMAND_EXPAND_LISTS
    VERBATIM
    )


    add_custom_command(TARGET IndustrialVision POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OpenCV_WORLD_DLLS}
            "$<TARGET_FILE_DIR:IndustrialVision>"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_command(TARGET ModelProbe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OpenCV_WORLD_DLLS}
        "$<TARGET_FILE_DIR:ModelProbe>"
    COMMAND_EXPAND_LISTS
    VERBATIM
)

endif()

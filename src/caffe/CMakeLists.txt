# generate protobuf sources
file(GLOB proto_files proto/*.proto)
caffe_protobuf_generate_cpp_py(${proto_gen_folder} proto_srcs proto_hdrs proto_python ${proto_files})

# include python files either to force generation
add_library(proto STATIC ${proto_hdrs} ${proto_srcs} ${proto_python})
set(Caffe_LINKER_LIBS proto ${Caffe_LINKER_LIBS}) # note, crucial to prepend!
caffe_default_properties(proto)

message(INFO, "Caffe_LINKER_LIBS")
message(INFO, "Caffe_LINKER_LIBS")
message(INFO, "Caffe_LINKER_LIBS")
message(INFO, ${Caffe_LINKER_LIBS})
message(INFO, "Caffe_LINKER_LIBS")
message(INFO, "Caffe_LINKER_LIBS")
message(INFO, "Caffe_LINKER_LIBS")
message(INFO, "Caffe_LINKER_LIBS")

# --[ Caffe library


file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


if(ANDROID)

file(GLOB_RECURSE jni-srcs ${CMAKE_CURRENT_SOURCE_DIR}/../jni/*.cpp)
add_library(caffe SHARED ${srcs} ${jni-srcs})
target_link_libraries(caffe proto ${Caffe_LINKER_LIBS} log)

else()
add_library(caffe STATIC ${srcs})
target_link_libraries(caffe proto ${Caffe_LINKER_LIBS})
endif()

caffe_default_properties(caffe)

#INSTALL(TARGETS caffe DESTINATION lib)

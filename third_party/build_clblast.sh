export NDK_HOME="/home/zrji/android_caffe/tmp_ndk/android-ndk-r18b"


MAKE_FLAGS="$MAKE_FLAGS -j 40"
BUILD_DIR=".cbuild"

# Options for Android
if [ "$ANDROID_ABI" = "" ]; then
  ANDROID_ABI="arm64-v8a"
fi

if [ "$ANDROID_NATIVE_API_LEVEL" = "" ]; then
  ANDROID_NATIVE_API_LEVEL=28
fi

if [ $ANDROID_NATIVE_API_LEVEL -lt 28 -a "$ANDROID_ABI" = "arm64-v8a" ]; then
    echo "ERROR: This ANDROID_ABI($ANDROID_ABI) requires ANDROID_NATIVE_API_LEVEL($ANDROID_NATIVE_API_LEVEL) >= 28"
    exit 1
fi




RUN_DIR=$PWD


function build-Linux {
    echo "$(tput setaf 2)"
    echo "#####################"
    echo " Building protobuf for Linux"
    echo "#####################"
    echo "$(tput sgr0)"

    mkdir -p CLBlast/$BUILD_DIR
    rm -rf CLBlast/$BUILD_DIR/*
    cd CLBlast/$BUILD_DIR
    if [ ! -s $Linux-CLBlast/lib/libCLBlast.a ]; then
        cmake ../cmake -DCMAKE_INSTALL_PREFIX=../../$Linux-protobuf \
            -Dprotobuf_BUILD_TESTS=OFF \
            -Dprotobuf_BUILD_SHARED_LIBS=OFF \
            -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
            -Dprotobuf_WITH_ZLIB=OFF
        make ${MAKE_FLAGS}
        make install
    fi
    cd ../..
    rm -f protobuf
    ln -s $Linux-protobuf protobuf
}

function build-MacOSX {
    build-Linux
}

function build-Android {
    TARGET="${ANDROID_ABI%% *}-$ANDROID_NATIVE_API_LEVEL"
    echo "$(tput setaf 2)"
    echo "#####################"
    echo " Building protobuf for $TARGET"
    echo "#####################"
    echo "$(tput sgr0)"

    # Test ENV NDK_HOME
    if [ ! -d "$NDK_HOME" ]; then
        echo "$(tput setaf 2)"
        echo "###########################################################"
        echo " ERROR: Invalid NDK_HOME=\"$NDK_HOME\" env variable, exit. "
        echo "###########################################################"
        echo "$(tput sgr0)"
        exit 1
    fi

    if [ ! -s ${TARGET}-CLBlast/lib/libCLBlast.a ]; then
        mkdir -p CLBlast/$BUILD_DIR
        rm -rf CLBlast/$BUILD_DIR/*
        cd CLBlast/$BUILD_DIR

        
        # cmake .. -DCMAKE_INSTALL_PREFIX=../../${TARGET}-CLBlast \
        #     -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
        #     -DANDROID_NDK="$NDK_HOME" \
        #     -DANDROID_ABI="$ANDROID_ABI" \
        #     -DANDROID_NATIVE_API_LEVEL="$ANDROID_NATIVE_API_LEVEL" \
        #     -Dprotobuf_BUILD_TESTS=OFF \
        #     -Dprotobuf_BUILD_SHARED_LIBS=OFF \
        #     -Dprotobuf_WITH_ZLIB=OFF \
        #     -DLDFLAGS="-llog" \
        #     -G "Unix Makefiles"


        cmake .. -DCMAKE_INSTALL_PREFIX=../../${TARGET}-CLBlast \
            -DCMAKE_SYSTEM_NAME=Android \
            -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
            -DCMAKE_SYSTEM_VERSION=28 \
            -DCMAKE_ANDROID_ARCH_ABI="$ANDROID_ABI" \
            -DANDROID_ABI="$ANDROID_ABI" \
            -DANDROID_NATIVE_API_LEVEL="$ANDROID_NATIVE_API_LEVEL" \
            -DCMAKE_ANDROID_NDK="$NDK_HOME" \
            -DCMAKE_ANDROID_STL_TYPE=gnustl_static \
            -DOPENCL_ROOT=/home/zrji/android_caffe/caffe-android-opencl/third_party/OpenCL




        make ${MAKE_FLAGS}
        make install
        cd ../..
    fi
}



# build-Linux
build-Android

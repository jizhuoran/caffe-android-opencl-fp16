MAKE_FLAGS="$MAKE_FLAGS -j 40"
BUILD_DIR=".cbuild"

# Options for Android
if [ "$ANDROID_ABI" = "" ]; then
  ANDROID_ABI="arm64-v8a"
fi

if [ "$ANDROID_NATIVE_API_LEVEL" = "" ]; then
  ANDROID_NATIVE_API_LEVEL=28
fi

if [ $ANDROID_NATIVE_API_LEVEL -lt 23 -a "$ANDROID_ABI" = "arm64-v8a" ]; then
    echo "ERROR: This ANDROID_ABI($ANDROID_ABI) requires ANDROID_NATIVE_API_LEVEL($ANDROID_NATIVE_API_LEVEL) >= 23"
    exit 1
fi


CLBlast_VERSION="master"

function fetch-CLBlast {
    echo "$(tput setaf 2)"
    echo "##########################################"
    echo " Fetch CLBlast $CLBlast_VERSION from source."
    echo "##########################################"
    echo "$(tput sgr0)"

    if [ ! -f CLBlast-${CLBlast_VERSION}.zip ]; then
        curl -L https://github.com/jizhuoran/CLBlast/archive/${CLBlast_VERSION}.zip --output CLBlast-${CLBlast_VERSION}.zip
    fi
    if [ -d CLBlast-${CLBlast_VERSION} ]; then
        rm -rf CLBlast-${CLBlast_VERSION}
    fi
    unzip CLBlast-${CLBlast_VERSION}.zip
}


function build-Android {
    TARGET="${ANDROID_ABI%% *}-$ANDROID_NATIVE_API_LEVEL"
    echo "$(tput setaf 2)"
    echo "#####################"
    echo " Building CLBlast for $TARGET"
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


    if [ ! -d "$DEVICE_OPENCL_DIR" ]; then
        echo "$(tput setaf 2)"
        echo "###########################################################"
        echo " ERROR: Invalid DEVICE_OPENCL_DIR=\"$DEVICE_OPENCL_DIR\" env variable, exit. "
        echo "###########################################################"
        echo "$(tput sgr0)"
        exit 1
    fi



    if [ ! -s ${TARGET}-CLBlast/lib/libclblast.a ]; then
        mkdir -p CLBlast-${CLBlast_VERSION}/$BUILD_DIR
        rm -rf CLBlast-${CLBlast_VERSION}/$BUILD_DIR/*
        cd CLBlast-${CLBlast_VERSION}/$BUILD_DIR


        cmake .. -DCMAKE_INSTALL_PREFIX=../../${TARGET}-CLBlast \
            -DCMAKE_SYSTEM_NAME=Android \
            -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
            -DCMAKE_SYSTEM_VERSION=23 \
            -DCMAKE_ANDROID_ARCH_ABI="$ANDROID_ABI" \
            -DANDROID_ABI="$ANDROID_ABI" \
            -DANDROID_NATIVE_API_LEVEL="$ANDROID_NATIVE_API_LEVEL" \
            -DCMAKE_ANDROID_NDK="$NDK_HOME" \
            -DCMAKE_ANDROID_STL_TYPE=gnustl_static \
            -DBUILD_SHARED_LIBS=OFF \
            -DOPENCL_ROOT=$DEVICE_OPENCL_DIR
        make ${MAKE_FLAGS}
        make install
        cd ../..
    fi
}


fetch-CLBlast
build-Android

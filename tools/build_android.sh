#!/bin/bash
export NDK_HOME="/home/zrji/android_caffe/tmp_ndk/android-ndk-r18b"

if [ ! -d "$NDK_HOME" ]; then
    echo "$(tput setaf 2)"
    echo "###########################################################"
    echo " ERROR: Invalid NDK_HOME=\"$NDK_HOME\" env variable, exit. "
    echo "###########################################################"
    echo "$(tput sgr0)"
    exit 1
fi

ANDROID_ABIs=("arm64-v8a")

function build-abi {
    cd third_party
    ./build-protobuf-3.1.0.sh Android || exit 1
    #exit 1
    ./build-openblas.sh || exit 1
    #exit 1
    ./build-clblast.sh || exit 1


    mkdir ../build_${ANDROID_ABI%% *}
    cd ../build_${ANDROID_ABI%% *} || exit 1
    rm -rf *
    cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
        -DANDROID_NDK=$NDK_HOME \
        -DANDROID_ABI="$ANDROID_ABI" \
        -DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL \
        -G "Unix Makefiles" || exit 1
    make -j 40 || exit 1
    cd ../examples/android/CaffeSimple/app/
    mkdir -p libs/${ANDROID_ABI%% *}
    ln -sf ../../../../../../build_${ANDROID_ABI%% *}/lib/libcaffe-jni.so libs/${ANDROID_ABI%% *}/libcaffe-jni.so
    cd ../../../..
}

IFS=""
for abi in ${ANDROID_ABIs[@]}; do
    export ANDROID_ABI="$abi"
    if [ "$ANDROID_ABI" = "arm64-v8a" ]; then
        export ANDROID_NATIVE_API_LEVEL=28
    else
        export ANDROID_NATIVE_API_LEVEL=16
    fi
    echo $ANDROID_ABI
    echo $ANDROID_NATIVE_API_LEVEL
    build-abi
done


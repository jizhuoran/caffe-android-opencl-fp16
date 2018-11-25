package com.example.gsq.caffe_android_project;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import com.example.gsq.caffe_android_project.CaffeMobile;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    static private String TAG = "MainActivity";
    private CaffeMobile _cm;


    int image_size = 512;


    protected class Cate implements Comparable<Cate> {
        public final int    idx;
        public final float  prob;

        public Cate(int idx, float prob) {
            this.idx = idx;
            this.prob = prob;
        }

        @Override
        public int compareTo(Cate other) {
            // need descending sort order
            return Float.compare(other.prob, this.prob);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 0: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.

                } else {

                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                }
                return;
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView tv = (TextView) findViewById(R.id.Console);
        tv.append("Loading caffe model...");
        tv.setMovementMethod(new ScrollingMovementMethod());
        // Show test image
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},0);
        }
        final File imageFile = new File(Environment.getExternalStorageDirectory(), "caffe/HKU.jpg");
        Bitmap bmp = BitmapFactory.decodeFile(imageFile.getPath());
        ImageView img = (ImageView) findViewById(R.id.testImage);
        img.setImageBitmap(bmp);

        // Load caffe model
        long start_time = System.nanoTime();
        File modelFile = new File(Environment.getExternalStorageDirectory(), "caffe/style.protobin");
        File weightFile = new File(Environment.getExternalStorageDirectory(), "caffe/a1.caffemodel");
        Log.d(TAG, "onCreate: modelFile:" + modelFile.getPath());
        Log.d(TAG, "onCreate: weightFIle:" + weightFile.getPath());
        _cm = new CaffeMobile();

        boolean res = _cm.loadModel(modelFile.getPath(), weightFile.getPath());

        long end_time = System.nanoTime();
        double difference = (end_time - start_time)/1e6;
        Log.d(TAG, "onCreate: loadmodel:" + res);
        tv.append(String.valueOf(difference) + "ms\n");

        //_cm.setBlasThreadNum(2);
        Button btn = (Button) findViewById(R.id.button);


        btn.setOnClickListener(new View.OnClickListener() {
            // Run test
            @Override
            public void onClick(View view) {
                final TextView tv = (TextView) findViewById(R.id.Console);
                tv.append("\nCaffe inferring...\n");
                final Handler myHandler = new Handler(){
                    @Override
                    public void handleMessage(Message msg) {

                        Bitmap bitmap = Bitmap.createBitmap(image_size, image_size, Bitmap.Config.ARGB_4444);
                        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap((byte [])msg.obj));
                        ImageView img = (ImageView) findViewById(R.id.testImage);

                        Matrix matrix = new Matrix();
                        matrix.postRotate(90);
                        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
                        img.setImageBitmap(rotatedBitmap);

                        Button btn = (Button) findViewById(R.id.button);
                        btn.setEnabled(true);


                    }
                };
                (new Thread(new Runnable() {
                    @Override
                    public void run() {
                        Message msg = myHandler.obtainMessage();
                        long start_time = System.nanoTime();
                        float mean[] = {81.3f, 107.3f, 105.3f};

                        float[] result = _cm.predictImage(imageFile.getPath(), mean);

                        long end_time = System.nanoTime();

                        byte [] picture = new byte [4*image_size*image_size];


                        if (null != result) {
                            double difference = (end_time - start_time) / 1e6;
                            // Top 10

                            for (int i = 0; i < image_size * image_size; i++) {
                                picture[4*i] = (byte) result[i];
                                picture[4*i+1] =  (byte) result[i + image_size * image_size];
                                picture[4*i+2] =  (byte) result[i + 2 * image_size * image_size];
                                picture[4*i+3] = (byte) 255;
                            }

                            msg.obj = picture;

                            Log.i("easy found", "The time used is " + String.valueOf(difference) + "ms):\n");
                        } else {
                        }
                        myHandler.sendMessage(msg);
                    }
                })).start();
            }
        });
    }

    // Used to load the 'caffe-jni' library on application startup.
    static {
        System.loadLibrary("caffe");
    }
}

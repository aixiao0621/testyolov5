package com.example.yolov5tfliteandroid;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;


import android.Manifest;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.widget.AdapterView;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import com.example.yolov5tfliteandroid.analysis.FullImageAnalyse;
import com.example.yolov5tfliteandroid.analysis.FullScreenAnalyse;
import com.example.yolov5tfliteandroid.detector.Yolov5TFLiteDetector;
import com.example.yolov5tfliteandroid.helper.FileHelper;
import com.example.yolov5tfliteandroid.holder.FileAddressHolder;
import com.example.yolov5tfliteandroid.userSelected.InputSize;
import com.example.yolov5tfliteandroid.userSelected.ModelAndLabel;
import com.example.yolov5tfliteandroid.userSelected.OutputSize;
import com.example.yolov5tfliteandroid.holder.TarExtractor;
import com.example.yolov5tfliteandroid.utils.CameraProcess;
import com.example.yolov5tfliteandroid.utils.FileUtils;
import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;

import java.io.FileNotFoundException;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

//    private final Size INPNUT_SIZE = new Size(320, 320);
//    private final int[] OUTPUT_SIZE = new int[]{1, 6300, 85};

    private boolean selectedModel=false;
    private boolean IS_FULL_SCREEN = false;

    private PreviewView cameraPreviewMatch;
    private PreviewView cameraPreviewWrap;
    private ImageView boxLabelCanvas;
    private Spinner modelSpinner;
    private Switch immersive;
    private TextView inferenceTimeTextView;
    private TextView frameSizeTextView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private Yolov5TFLiteDetector yolov5TFLiteDetector;
    private TextView file_btn;
    private static final int PICK_FILE_REQUEST_CODE = 1; // 请求代码，用于标识文件选择器的返回结果
    private TextView show_address;


    private CameraProcess cameraProcess = new CameraProcess();
    private Size UserSelected_INPUT_SIZE;
    private  int[] UserSelected_OUTPUT_SIZE;
    private int rotation;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;


    @Override
    public void onResume() {
        super.onResume();
        Log.e("resume","restart");
    }


    /**
     * 获取屏幕旋转角度,0表示拍照出来的图片是横屏
     *
     */
    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    /**
     * 加载模型
     *
     * @param modelName
     */
    private void initModel(String modelName,String labelFile,Size inputsize,int[] outputsize) {
        // 加载模型
        try {
            this.yolov5TFLiteDetector = new Yolov5TFLiteDetector();
            Log.d("gz", "111");
            this.yolov5TFLiteDetector.setModelFile(modelName);
            Log.d("gz", "222");
//            this.yolov5TFLiteDetector.addNNApiDelegate();
            this.yolov5TFLiteDetector.addGPUDelegate();
            Log.d("gz", "333");
            this.yolov5TFLiteDetector.initialModel(this,labelFile,inputsize,outputsize);
            Log.i("model", "Success loading model" + this.yolov5TFLiteDetector.getModelFile());
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(MainActivity.this,"Invalid file format selected! Please choose again.",Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 打开app的时候隐藏顶部状态栏
//        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
        getWindow().setStatusBarColor(Color.TRANSPARENT);

        // 全屏画面
        cameraPreviewMatch = findViewById(R.id.camera_preview_match);
        cameraPreviewMatch.setScaleType(PreviewView.ScaleType.FILL_START);

        // 全图画面
        cameraPreviewWrap = findViewById(R.id.camera_preview_wrap);
//        cameraPreviewWrap.setScaleType(PreviewView.ScaleType.FILL_START);

        // box/label画面
        boxLabelCanvas = findViewById(R.id.box_label_canvas);

//        // 下拉按钮
//        modelSpinner = findViewById(R.id.model);

        // 沉浸式体验按钮
        immersive = findViewById(R.id.immersive);

        // 实时更新的一些view
        inferenceTimeTextView = findViewById(R.id.inference_time);
        frameSizeTextView = findViewById(R.id.frame_size);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        // 申请摄像头权限
        if (!cameraProcess.allPermissionsGranted(this)) {
            cameraProcess.requestPermissions(this);
        }

        // 获取手机摄像头拍照旋转参数
        rotation = getWindowManager().getDefaultDisplay().getRotation();
        Log.i("image", "rotation: " + rotation);

        cameraProcess.showCameraSupportSize(MainActivity.this);

        // 初始化加载,目前没有初始化加载模型
//        initModel("yolov5s","coco_label.txt",INPNUT_SIZE,OUTPUT_SIZE);//位置？

        //打开用户本地文件按钮
        file_btn=findViewById(R.id.file_btn);


        file_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    // 如果没有权限，请求权限
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_EXTERNAL_STORAGE);
                } else {
                    // 已经有权限，执行文件访问的操作
                    // 在这里调用显示文件选择器的方法
                    showFileChooser();
                }

            }
        });

        // 监听视图变化按钮
        immersive.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                if (selectedModel == true) {
                    Log.e("test", "b：" + b);
                    IS_FULL_SCREEN = b;
                    if (b) {
                        // 进入全屏模式
                        Log.e("test", "全屏模式");
                        cameraPreviewWrap.removeAllViews();
                        FullScreenAnalyse fullScreenAnalyse = new FullScreenAnalyse(MainActivity.this,
                                cameraPreviewMatch,
                                boxLabelCanvas,
                                rotation,
                                inferenceTimeTextView,
                                frameSizeTextView,
                                UserSelected_INPUT_SIZE,
                                UserSelected_OUTPUT_SIZE,
                                yolov5TFLiteDetector);
                        cameraProcess.startCamera(MainActivity.this, fullScreenAnalyse, cameraPreviewMatch);

                    } else {
                        // 进入全图模式
                        Log.e("test", "全图模式");
                        //TODO
                        cameraPreviewMatch.removeAllViews();
                        FullImageAnalyse fullImageAnalyse = new FullImageAnalyse(
                                MainActivity.this,
                                cameraPreviewWrap,
                                boxLabelCanvas,
                                rotation,
                                inferenceTimeTextView,
                                frameSizeTextView,
                                UserSelected_INPUT_SIZE,
                                UserSelected_OUTPUT_SIZE,
                                yolov5TFLiteDetector);
                        cameraProcess.startCamera(MainActivity.this, fullImageAnalyse, cameraPreviewWrap);
                    }
                }
                else {
                    Toast.makeText(MainActivity.this,"请先选择一个模型",Toast.LENGTH_SHORT).show();
                }
            }
        });

    }
         private void showFileChooser() {

            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);//不限制用户选择的文件内容
            intent.setType("*/*"); // 设置文件类型，可以根据需要修改
            intent.addCategory(Intent.CATEGORY_OPENABLE);

        try {
            startActivityForResult(
                    Intent.createChooser(intent, "选择一个文件"),
                    PICK_FILE_REQUEST_CODE);
            Log.e("test", "选择到一个文件");
        } catch (android.content.ActivityNotFoundException ex) {
            // 如果用户没有安装文件管理器应用，可以添加适当的处理
        }
    }

    String path;
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK) {
            Uri uri = data.getData();

            byte[] fileBytes = FileHelper.getFileBytesFromUri(this, uri);

            // 处理文件内容字节数组
            if (fileBytes != null) {
                String filePath = saveBytesToFile(this, fileBytes, "YourFileName.extension");
                if (filePath != null) {
                    Log.d("FilePath", filePath);
                    // filePath 即为保存的文件路径
                    file_btn.setText("fp16");
            // 提取Tar文件的操作放在后台线程中执行
            ExtractTask extractTask = new ExtractTask();
            extractTask.execute(filePath);

                }else{
                    //添加选择错误判定
                }
            }else{
                //添加选择错误判定
            }

        }

    }

    public static String saveBytesToFile(Context context, byte[] fileBytes, String fileName) {
        if (fileBytes == null || fileName == null) {
            return null;
        }

        try {
            File dir = new File(context.getExternalFilesDir(null) + File.separator + "YourDirectoryName");
            if (!dir.exists()) {
                dir.mkdir();
            }

            File file = new File(dir, fileName);
            FileOutputStream outputStream = new FileOutputStream(file);
            outputStream.write(fileBytes);
            outputStream.close();

            return file.getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }


    private class ExtractTask extends AsyncTask<String, Void, Void> {

    @Override
    protected Void doInBackground(String... filePaths) {
        // 在后台进行解压操作
        if (filePaths != null && filePaths.length > 0) {
            String filePath = filePaths[0];
            JNITools test = new JNITools();
            byte[] input = test.extractTarGz(filePath);
            File outputDirectory = null;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                outputDirectory = getApplicationContext().getDataDir();
            }
            TarExtractor tarExtractor = new TarExtractor();
            try {
                tarExtractor.extractTar(input, outputDirectory);
            } catch (IOException e) {
                e.printStackTrace();
            }
//            writeFile("test",input);
        }
        return null;
    }

        @Override
        protected void onPostExecute(Void result) {
            File outputDirectory = null;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
                outputDirectory = getApplicationContext().getDataDir();

                // 把流文件提取出来的信息，显示出来，存储在单例全局变量中
                String modelDirectory = null;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                    modelDirectory = getApplicationContext().getDataDir().toString();
                } else {
                    modelDirectory = getApplicationContext().getExternalFilesDir(null).toString() + "/yolo";
                }
                String jsonFileAddress = modelDirectory + "/config.json";
                Log.e("TNT", modelDirectory + "/config.json");

                ModelAndLabel.getInstance().setLabel(modelDirectory + "/label.txt");

                Log.e("TNT",ModelAndLabel.getInstance().getLabel());

                String filePath = ModelAndLabel.getInstance().getLabel();
                // 读取文件内容并打印
                try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                    String line;
                    StringBuilder content = new StringBuilder();
                    while ((line = br.readLine()) != null) {
                        content.append(line).append("\n");
                    }
                    Log.e("test", content.toString());
                } catch (IOException e) {
                    e.printStackTrace();
                }

                try {
                    String json = readJsonFile(jsonFileAddress);
                    Log.e("test", json);
                    try {
                        Log.e("test", "开始解析json");
                        JSONObject jsonObject = new JSONObject(json);

                        // 只获取 output_size 的数字
                        JSONArray outputSizeArray = jsonObject.getJSONArray("output_size");
                        StringBuilder outputSizeStringBuilder = new StringBuilder();
                        for (int i = 0; i < outputSizeArray.length(); i++) {
                            outputSizeStringBuilder.append(outputSizeArray.getInt(i));
                            if (i < outputSizeArray.length()) {
                                Log.e("test", String.valueOf(outputSizeArray.getInt(i)));
                                if (i == 0) {
                                    OutputSize.getInstance().setOutput_Param1(outputSizeArray.getInt(i));
                                }
                                else if (i == 1) {
                                    OutputSize.getInstance().setOutput_Param2(outputSizeArray.getInt(i));
                                }
                                else if (i == 2) {
                                    OutputSize.getInstance().setOutput_Param3(outputSizeArray.getInt(i));
                                }
                            }
                        }
                        // 只获取 input_size 的数字
                        JSONArray inputSizeArray = jsonObject.getJSONArray("input_size");
                        StringBuilder inputSizeStringBuilder = new StringBuilder();
                        for (int i = 0; i < inputSizeArray.length(); i++) {
                            inputSizeStringBuilder.append(inputSizeArray.getInt(i));
                            if (i < inputSizeArray.length()) {
                                Log.e("test", String.valueOf(inputSizeArray.getInt(i)));
                                // 根据索引 i 设置对应的输入参数值
                                if (i == 0) {
                                    InputSize.getInstance().setInput_Param1(inputSizeArray.getInt(i));
                                } else if (i == 1) {
                                    InputSize.getInstance().setInput_Param2(inputSizeArray.getInt(i));
                                }
                            }
                        }
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                ModelAndLabel.getInstance().setModel(modelDirectory + "/fp16.tflite");
                ModelAndLabel.getInstance().setLabel(modelDirectory + "/label.txt");

                Log.e("test", ModelAndLabel.getInstance().getModel());
                Log.e("test", ModelAndLabel.getInstance().getLabel());

                Toast.makeText(MainActivity.this, "loading model:" + ModelAndLabel.getInstance().getModel(), Toast.LENGTH_LONG).show();

                rotation = getWindowManager().getDefaultDisplay().getRotation();

                UserSelected_INPUT_SIZE = new Size(InputSize.getInstance().getInput_Param1(), InputSize.getInstance().getInput_Param2());
                UserSelected_OUTPUT_SIZE = new int[]{OutputSize.getInstance().getOutput_Param1(), OutputSize.getInstance().getOutput_Param2(), OutputSize.getInstance().getOutput_Param3()};

                initModel("UserSelected", ModelAndLabel.getInstance().getLabel(), UserSelected_INPUT_SIZE, UserSelected_OUTPUT_SIZE);
                selectedModel=true;
//                if(testBitmap!=null){
//                    Log.e("MMMM2", "testBitmap不为空");
//                    bitmap_test.setImageBitmap(testBitmap);
//                }
                if (IS_FULL_SCREEN) {
                    cameraPreviewWrap.removeAllViews();
                    FullScreenAnalyse fullScreenAnalyse = new FullScreenAnalyse(MainActivity.this,
                            cameraPreviewMatch,
                            boxLabelCanvas,
                            rotation,
                            inferenceTimeTextView,
                            frameSizeTextView,
                            UserSelected_INPUT_SIZE,
                            UserSelected_OUTPUT_SIZE,
                            yolov5TFLiteDetector);
                    cameraProcess.startCamera(MainActivity.this, fullScreenAnalyse, cameraPreviewMatch);
                } else {
                    cameraPreviewMatch.removeAllViews();
                    FullImageAnalyse fullImageAnalyse = new FullImageAnalyse(
                            MainActivity.this,
                            cameraPreviewWrap,
                            boxLabelCanvas,
                            rotation,
                            inferenceTimeTextView,
                            frameSizeTextView,
                            UserSelected_INPUT_SIZE,
                            UserSelected_OUTPUT_SIZE,
                            yolov5TFLiteDetector);
                    cameraProcess.startCamera(MainActivity.this, fullImageAnalyse, cameraPreviewWrap);
                }

            }
        }
    }


        private static String readJsonFile(String filePath) throws IOException {
            Path path = null;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                path = Paths.get(filePath);
            }
            byte[] bytes = new byte[0];
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                bytes = Files.readAllBytes(path);
            }
            return new String(bytes);
        }

        public void writeFile(String fileName, byte[] content) {
            Context context = getApplicationContext();
            try (FileOutputStream fos = context.openFileOutput(fileName, Context.MODE_PRIVATE)) {
                fos.write(content);
            } catch (FileNotFoundException e) {
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
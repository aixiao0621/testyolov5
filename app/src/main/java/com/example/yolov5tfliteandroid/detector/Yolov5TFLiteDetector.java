package com.example.yolov5tfliteandroid.detector;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Build;
import android.util.Log;
import android.util.Size;
import android.widget.Toast;

import com.example.yolov5tfliteandroid.userSelected.InputSize;
import com.example.yolov5tfliteandroid.userSelected.ModelAndLabel;
import com.example.yolov5tfliteandroid.userSelected.OutputSize;
import com.example.yolov5tfliteandroid.utils.Recognition;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import com.example.yolov5tfliteandroid.MainActivity;

public class Yolov5TFLiteDetector {

     public native long createDetector(byte[] modelBuffer, long size);
    public native float[] detect(long detectorPtr,
                                 byte[] src,
                                 int width,
                                 int height);
    class box_t{
        public float x1, y1, x2, y2, score, label;
        public box_t(float x1, float y1, float x2, float y2, float score, float label){
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
            this.label = label;
        }
    }

//    private final Size INPNUT_SIZE = new Size(320, 320);
//    private final int[] OUTPUT_SIZE = new int[]{1, 6300, 85};

    private Boolean IS_INT8 = false;
//    private final float DETECT_THRESHOLD = 0.0f;
//    private final float IOU_THRESHOLD = 0.0f;
//    private final float IOU_CLASS_DUPLICATED_THRESHOLD = 0.0f;
    private final float IOU_CLASS_DUPLICATED_THRESHOLD = 0.7f;
    private final float IOU_THRESHOLD = 0.45f;
    private final float DETECT_THRESHOLD = 0.25f;

    private Size UserSelected_INPUT_SIZE = new Size(InputSize.getInstance().getInput_Param1(), InputSize.getInstance().getInput_Param2());
    private int[] UserSelected_OUTPUT_SIZE = new int[]{OutputSize.getInstance().getOutput_Param1(), OutputSize.getInstance().getOutput_Param2(), OutputSize.getInstance().getOutput_Param3()};
    private String UserSelected_MODEL = ModelAndLabel.getInstance().getModel();
    private String UserSelected_LABEL_FILE = ModelAndLabel.getInstance().getLabel();
    MetadataExtractor.QuantizationParams input5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.003921568859368563f, 0);
    MetadataExtractor.QuantizationParams output5SINT8QuantParams = new MetadataExtractor.QuantizationParams(0.006305381190031767f, 5);
    private String MODEL_FILE;
    private List<String> associatedAxisLabels;
    private Size Final_inputSize;
    private  int[] Final_outputSize;

    private long detectorPtr;

    private float[] outputBuffer;

    private ArrayList<box_t> boxes = new ArrayList<>();


    Interpreter.Options options = new Interpreter.Options();

    public Yolov5TFLiteDetector() {
    }

    public String getModelFile() {
        return this.MODEL_FILE;
    }

    public void setModelFile(String modelFile) {
        switch (modelFile) {
            case "UserSelected":
                IS_INT8 = false;
                MODEL_FILE = UserSelected_MODEL;
                Log.e("test","加载用户模型（选择）");
                break;
            default:
                Log.e("test","default");
                Log.i("tfliteSupport", "Only yolov5s/n/m/sint8 can be load!");
        }
    }

//    public String getLabelFile() {
//        return this.LABEL_FILE;
//    }

    public int getInputSizeWidth() {
        return InputSize.getInstance().getInput_Param1();
    }
    public int getInputSizeHeight() {
        return InputSize.getInstance().getInput_Param2();
    }

//    public int[] getOutputSize() {
//        return this.OUTPUT_SIZE;
//    }

    /**
     * 初始化模型, 可以通过 addNNApiDelegate(), addGPUDelegate()提前加载相应代理
     *
     * @param activity
     */
    public void initialModel(Context activity, String labelFile,Size inputsize,int[] outputsize) {
        // Initialise the model
        Final_inputSize = inputsize;
        Final_outputSize = outputsize;
        Log.e("test-user", "用户选择模型正常运行");
        ByteBuffer tfliteModel = null;

        // 获取模型文件的路径
        String modelFilePath = ModelAndLabel.getInstance().getModel();
        File modelFile = new File(modelFilePath);
        Log.e("aabb", modelFile.getAbsolutePath());

        try {
            FileInputStream fis = new FileInputStream(modelFile);
            byte[] modelBytes = new byte[(int) modelFile.length()];
            fis.read(modelBytes);
            detectorPtr = createDetector(modelBytes, modelBytes.length);
            Log.i("tfliteSupport", "Success reading model: " + modelFilePath);
        } catch (IOException e) {
            e.printStackTrace();
            Log.d("aabb", "readfile failed");
            Log.e("tfliteSupport", "Error reading model: " + e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            Log.d("aabb", "-->" + e);
            Log.e("tfliteSupport", "Error reading model: ===>" + e);
        }

        // 获取标签文件的路径
        String labelFilePath = ModelAndLabel.getInstance().getLabel();

        try {
            File labelFileFile = new File(labelFilePath);
            associatedAxisLabels = FileUtil.loadLabels(new FileInputStream(labelFileFile));

            Log.i("tfliteSupport", "Success reading label: " + labelFilePath);
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("tfliteSupport", "Error reading label: " + e.getMessage());
        }

    }

    /**
     * 检测步骤
     *
     * @param bitmap
     * @return
     */
    public ArrayList<Recognition> detect(Bitmap bitmap) {
        ArrayList<Recognition> allRecognitions = new ArrayList<>();
        // flatten image >> detect
        // 检查传入的位图是否为空
        if (bitmap == null) {
            Log.e("test", "Bitmap is null. Unable to create TensorImage.");
            return new ArrayList<>();
        }

        // 创建 TensorImage 以加载位图
        TensorImage yolov5sTfliteInput;

        yolov5sTfliteInput = new TensorImage(DataType.UINT8);
        // 加载位图到 TensorImage
        yolov5sTfliteInput.load(bitmap);
        if (yolov5sTfliteInput == null) {
            Log.e("test", "TensorImage is null after loading bitmap. Unable to continue detection.");
            return new ArrayList<>();
        }

        // yolov5s-tflite的输出是:[1, 6300, 85], 可以从v5的GitHub release处找到相关tflite模型, 输出是[0,1], 处理到320.
        TensorBuffer probabilityBuffer;
        Log.d("fucked", "data" + yolov5sTfliteInput.getBuffer().array().length);
        // 推理计算
        if (detectorPtr != 0) {
            outputBuffer = detect(
                    detectorPtr,
                    yolov5sTfliteInput.getTensorBuffer().getBuffer().array(),
                    yolov5sTfliteInput.getWidth(),
                    yolov5sTfliteInput.getHeight());
            if (outputBuffer.length != 0){
                boxes.clear();

                for (int i = 0; i < outputBuffer.length; i += 6){
                    boxes.add(new box_t(
                            outputBuffer[i],
                            outputBuffer[i+1],
                            outputBuffer[i+2],
                            outputBuffer[i+3],
                            outputBuffer[i+4],
                            outputBuffer[i+5])
                    );
                }
                Log.d("fuck", "==?" +
                        boxes.get(0).x1 + "?" +
                        boxes.get(0).y1 + "?" +
                        boxes.get(0).x2 + "?" +
                        boxes.get(0).y2 + ">" +
                        boxes.get(0).score + "?" +
                        associatedAxisLabels.get((int) boxes.get(0).label));
            }

        }

        Log.d("fuck", "" + boxes.size());
        for (int i = 0; i < boxes.size(); i++) {
            box_t box = boxes.get(i);
            if(box.label > associatedAxisLabels.size()){
                Log.d("fuck", "label > associatedAxisLabels.size()" + box.label);
                continue;
            }
            Recognition r = new Recognition(
                    (int) box.label,
                    associatedAxisLabels.get((int) box.label),
                    box.score,
                    box.score,
                    new RectF(box.x1, box.y1, box.x2, box.y2));
            allRecognitions.add(r);
        }

        return allRecognitions;
    }

    protected float boxIou(RectF a, RectF b) {
        float intersection = boxIntersection(a, b);
        float union = boxUnion(a, b);
        if (union <= 0) return 1;
        return intersection / union;
    }

    protected float boxIntersection(RectF a, RectF b) {
        float maxLeft = a.left > b.left ? a.left : b.left;
        float maxTop = a.top > b.top ? a.top : b.top;
        float minRight = a.right < b.right ? a.right : b.right;
        float minBottom = a.bottom < b.bottom ? a.bottom : b.bottom;
        float w = minRight -  maxLeft;
        float h = minBottom - maxTop;

        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float boxUnion(RectF a, RectF b) {
        float i = boxIntersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    /**
     * 添加NNapi代理
     */
    public void addNNApiDelegate() {
        NnApiDelegate nnApiDelegate = null;
        // Initialize interpreter with NNAPI delegate for Android Pie or above
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {

//            NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
//            nnApiOptions.setAllowFp16(true);
//            nnApiOptions.setUseNnapiCpu(true);
//            ANEURALNETWORKS_PREFER_LOW_POWER：//倾向于以最大限度减少电池消耗的方式执行。这种设置适合经常执行的编译。
//            ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER：//倾向于尽快返回单个答案，即使这会耗费更多电量。这是默认值。
//            ANEURALNETWORKS_PREFER_SUSTAINED_SPEED：//倾向于最大限度地提高连续帧的吞吐量，例如，在处理来自相机的连续帧时。
//            nnApiOptions.setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED);
//            nnApiDelegate = new NnApiDelegate(nnApiOptions);

            nnApiDelegate = new NnApiDelegate();
            options.addDelegate(nnApiDelegate);
            Log.i("tfliteSupport", "using nnapi delegate.");
        }
    }

    /**
     * 添加GPU代理
     */
    public void addGPUDelegate() {
        CompatibilityList compatibilityList = new CompatibilityList();
        if(compatibilityList.isDelegateSupportedOnThisDevice()){
            GpuDelegate.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            options.addDelegate(gpuDelegate);
            Log.i("tfliteSupport", "using gpu delegate.");
        } else {
            addThread(4);
        }
    }

    /**
     * 添加线程数
     * @param thread
     */
    public void addThread(int thread) {
        options.setNumThreads(thread);
    }

}

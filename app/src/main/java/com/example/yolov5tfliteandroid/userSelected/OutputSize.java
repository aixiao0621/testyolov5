package com.example.yolov5tfliteandroid.userSelected;

public class OutputSize {
    private int width;
    private int height;
    private int depth;

    private int Output_Param1;
    private int Output_Param2;
    private int Output_Param3;
    private static OutputSize instance = new OutputSize();
    private OutputSize() {}
    public static OutputSize getInstance() {
        return instance;
    }



    // 提供 getter 和 setter 方法，允许随时修改
    public void setOutput_Param1(int Output_Param11) {
        this.Output_Param1 = Output_Param11;
    }
    public int getOutput_Param1() {
        return Output_Param1;
    }

    public void setOutput_Param2(int Output_Param22) {
        this.Output_Param2 = Output_Param22;
    }
    public int getOutput_Param2() {
        return Output_Param2;
    }

    public void setOutput_Param3(int Output_Param33) {
        this.Output_Param3 = Output_Param33;
    }
    public int getOutput_Param3() {
        return Output_Param3;
    }


}

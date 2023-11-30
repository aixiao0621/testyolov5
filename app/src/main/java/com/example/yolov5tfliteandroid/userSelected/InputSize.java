package com.example.yolov5tfliteandroid.userSelected;

import com.example.yolov5tfliteandroid.holder.FileAddressHolder;

public class InputSize {
    private int Input_Param1;
    private int Input_Param2;

    private static InputSize instance = new InputSize();
    private InputSize() {}
    public static InputSize getInstance() {
        return instance;
    }

    // 其他方法...

    public void setInput_Param1(int Input_Param11 ) {
        this.Input_Param1 = Input_Param11;
    }
    public int getInput_Param1() {
        return Input_Param1;
    }


    public void setInput_Param2(int input_Param22) {
        Input_Param2 = input_Param22;
    }

    public int getInput_Param2() {
        return Input_Param2;
    }

}


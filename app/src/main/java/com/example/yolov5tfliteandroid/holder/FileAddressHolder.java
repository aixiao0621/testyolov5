package com.example.yolov5tfliteandroid.holder;

public class FileAddressHolder {

    private static FileAddressHolder instance = new FileAddressHolder();
    private FileAddressHolder() {}
    public static FileAddressHolder getInstance() {
        return instance;
    }
    private String fileAddress;

    public void setFileAddress(String selectedFileAddress){
        fileAddress=selectedFileAddress;
    }
    public String getFileAddress(){
        return fileAddress;
    }

}

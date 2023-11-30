package com.example.yolov5tfliteandroid.userSelected;

public class ModelAndLabel {
    private String model;
    private String label;


    private static ModelAndLabel instance = new ModelAndLabel();
    private ModelAndLabel() {}
    public static ModelAndLabel getInstance() {
        return instance;
    }

    public ModelAndLabel(String model, String label) {
        this.model = model;
        this.label = label;
    }

    // Provide getter and setter methods for model
    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    // Provide getter and setter methods for label
    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }
}

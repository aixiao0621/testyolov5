package com.example.yolov5tfliteandroid.holder;

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import java.io.*;

public class TarExtractor {

    public  void extractTar(byte[] tarFileStream, File outputDirectory) throws IOException {
        try (TarArchiveInputStream tis = new TarArchiveInputStream(new ByteArrayInputStream(tarFileStream))) {
            TarArchiveEntry entry;
            while ((entry = tis.getNextTarEntry()) != null) {
                extractEntry(tis, entry, outputDirectory);
            }
        }
    }

    private static void extractEntry(TarArchiveInputStream tis, TarArchiveEntry entry, File outputDirectory) throws IOException {
        File outputFile = new File(outputDirectory, entry.getName());
        if (entry.isDirectory()) {
            if (!outputFile.exists() && !outputFile.mkdirs()) {
                throw new IOException("Failed to create directory: " + outputFile.getAbsolutePath());
            }
        } else {
            if (isFileInBestDirectory(entry, "best")) {
                String fileName = entry.getName().substring("best/".length());
                try (FileOutputStream fos = new FileOutputStream(new File(outputDirectory, fileName))) {
                    byte[] buffer = new byte[1024];
                    int read;
                    while ((read = tis.read(buffer)) != -1) {
                        fos.write(buffer, 0, read);
                    }
                }
            }
        }
    }

    private static boolean isFileInBestDirectory(TarArchiveEntry entry, String directory) {
        return entry.getName().startsWith(directory + "/");
    }

}


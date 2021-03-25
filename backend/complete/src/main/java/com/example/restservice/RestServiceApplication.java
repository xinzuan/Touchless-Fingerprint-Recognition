package com.example.restservice;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.example.restservice.models.entities.CandidatesCache;
import com.example.restservice.repositories.CandidatesRepository;

import org.apache.commons.io.FilenameUtils;
import org.springframework.beans.factory.annotation.Autowired;
import com.machinezoo.sourceafis.FingerprintTemplate;
import com.machinezoo.sourceafis.FingerprintImage;
import javax.annotation.PostConstruct;

import java.io.*;
import java.util.*;

import org.springframework.beans.factory.annotation.Value;


@SpringBootApplication
public class RestServiceApplication {

    @Autowired
    private CandidatesRepository candidatesRepository;
  
    // @Value("${dev.app.db}")
	private String db_path = "/home/vania/TA/Implement/Touchless-Fingerprint-Recognition/backend/complete/src/resources/fingerprints/";

    public static void main(String[] args) {
        SpringApplication.run(RestServiceApplication.class, args);
    }


    @PostConstruct
    private void initDb() {
        
     
        try {
            for (Path path : Files.newDirectoryStream(Paths.get(db_path), 
                        path -> path.toFile().isFile())) {
                
                FingerprintTemplate template = new FingerprintTemplate(
                    new FingerprintImage()
                        .dpi(500)
                        .decode(Files.readAllBytes(Paths.get(path.toString()))));
                String name = FilenameUtils.removeExtension(path.getFileName().toString());
                // template harus di serialized menggunakan toByteArray() sebelum disimpan 
              
                
                byte[] serialized = template.toByteArray();
             
                CandidatesCache candidate = new CandidatesCache(name,serialized);
           
                candidatesRepository.save(candidate);
              

            }
        } catch (IOException e) {
            //error
        }

  }

}

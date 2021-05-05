package com.example.restservice.services.impl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


import com.example.restservice.models.entities.User;
import com.example.restservice.models.entities.CandidatesCache;
import com.example.restservice.models.entities.CandidateDetails;
import com.example.restservice.models.inbounds.UserInbound;

// import com.example.restservice.repositories.UserRepository;
import com.example.restservice.repositories.CandidatesRepository;
import com.example.restservice.services.api.UserService;
import java.util.*; 
import java.io.*;

import com.machinezoo.sourceafis.FingerprintTemplate;
import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintMatcher;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;



import org.apache.commons.io.FilenameUtils;




@Service
public class UserServiceImpl implements UserService {

  // @Autowired
  // private UserRepository userRepository;

  @Autowired
  private CandidatesRepository candidatesRepository;


  @Override
  public User findMatch(UserInbound userInbound,List<CandidateDetails> candidates) {
   
    CandidateDetails match = null;
    double high = 0;

    // TODO : ADD FILE NOT FOUND EXCEPTION HANDLE
    try{
        FingerprintTemplate userTemplate = new FingerprintTemplate(
                      new FingerprintImage()
                          .dpi(500)
                          .decode(Files.readAllBytes(Paths.get(userInbound.getPathimage()))));
        FingerprintMatcher matcher = new FingerprintMatcher()
                  .index(userTemplate);



        for (CandidateDetails candidate : candidates) {
            
            double score = matcher.match(candidate.getTemplate());
            if (score > high) {
                high = score;
                match = candidate;
            }
        }

    	} catch (Exception e ){
			    e.printStackTrace();
			}
      
      double threshold = 39.9;
     
      if (high >= threshold){
        User user = new User(match.getName(),high);
        return user;
      } else {
        return null;
      }
      
        
   
  }
  @Override
  public CandidatesCache addUser(UserInbound userInbound) {
     
      String path = userInbound.getPathimage();
      // System.out.println(path);
      try{
          FingerprintTemplate template = new FingerprintTemplate(
                        new FingerprintImage()
                            .dpi(500)
                            .decode(Files.readAllBytes(Paths.get(path))));
          String name = userInbound.getName();
                    // template harus di serialized menggunakan toByteArray() sebelum disimpan 
                  
                    
          byte[] serialized = template.toByteArray();
        
          CandidatesCache candidate = new CandidatesCache(name,serialized);
          
          candidatesRepository.save(candidate);
          
          return candidate;
        } catch (Exception e ){
			    e.printStackTrace();
          return null;
			}
  }
}

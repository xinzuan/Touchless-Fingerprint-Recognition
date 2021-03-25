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
      
      double threshold = 40;
     
      if (high >= threshold){
        User user = new User(match.getName(),high);
        return user;
      } else {
        return new User("Unidentified",high);
      }
      
        
   
  }
  @Override
  public CandidatesCache addUser(UserInbound userInbound) {
        String fromFile = userInbound.getPathimage();
        String toFile = "/home/mkyong/data/deploy/db.conf";

        Path source = Paths.get(fromFile);
        Path target = Paths.get(toFile);

        try {

            // rename or move a file to other path
            // if target exists, throws FileAlreadyExistsException
            Files.move(source, target);

            // if target exists, replace it.
            // Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);

            // multiple CopyOption
            /*CopyOption[] options = { StandardCopyOption.REPLACE_EXISTING,
                                StandardCopyOption.COPY_ATTRIBUTES,
                                LinkOption.NOFOLLOW_LINKS };

            Files.move(source, target, options);*/

        } catch (IOException e) {
            e.printStackTrace();
        }
        CandidatesCache user = CandidatesCache.builder()
            .name(userInbound.getName())
            .build();
        candidatesRepository.save(user);
        return user;
  }
}

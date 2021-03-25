package com.example.restservice.services.api;

import com.example.restservice.models.entities.User;
import com.example.restservice.models.entities.CandidateDetails;
import com.example.restservice.models.entities.CandidatesCache;
import com.example.restservice.models.inbounds.UserInbound;
import java.util.*; 
public interface UserService {
 


  User findMatch(UserInbound user,List<CandidateDetails> candidates);
  CandidatesCache addUser(UserInbound user);


}

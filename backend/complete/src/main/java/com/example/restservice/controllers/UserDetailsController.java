package com.example.restservice.controllers;

import java.util.*;
import java.lang.Object;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.http.HttpStatus;

import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.beans.factory.annotation.Autowired;

import com.machinezoo.sourceafis.FingerprintTemplate;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintTransparency;

import com.example.restservice.models.inbounds.UserInbound;
import com.example.restservice.models.entities.User;
import com.example.restservice.services.api.UserService;
import com.example.restservice.models.entities.CandidateDetails;
import com.example.restservice.models.entities.CandidatesCache;
import com.example.restservice.repositories.CandidatesRepository;
import com.example.restservice.models.outbounds.wrapper.DataResponse;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


@RestController
public class UserDetailsController {




	@Autowired
	private UserService userService;
	@Autowired
	private CandidatesRepository candidatesRepository;

	@GetMapping("/match")
	public DataResponse<User> findMatch(@RequestBody UserInbound userInbound) {
		
		List<CandidateDetails> candidates = new ArrayList<CandidateDetails>();
		List<CandidatesCache> candidatesR = candidatesRepository.findAll();
		long id = 0;
		for( CandidatesCache element : candidatesR ){ 
			FingerprintTemplate template = new FingerprintTemplate(element.getTemplate());
			CandidateDetails cd = new CandidateDetails(id,element.getName(),template);
			candidates.add(cd);
			id++;
		}
		return DataResponse.<User>dataBuilder()
			.data(userService.findMatch(userInbound,candidates))
			.status(HttpStatus.OK.getReasonPhrase())
			.code(HttpStatus.OK.value())
			.build();

			
		}

	@PostMapping("/try")
	public String tryAPI(@RequestBody UserInbound userInbound) {
		System.out.println("stare..");
		return userInbound.getPathimage();

			
		}
	@PostMapping
  	public DataResponse<CandidatesCache> createUser(
      @RequestBody
          UserInbound userInbound
  ) {
    return DataResponse.<CandidatesCache>dataBuilder()
        .data(userService.addUser(userInbound))
        .status(HttpStatus.OK.getReasonPhrase())
        .code(HttpStatus.OK.value())
        .build();
  }
}

// 		String res ="Not matched";
// try (
//     OutputStream stream = new FileOutputStream("transparency.zip");
//     FingerprintTransparency transparency = FingerprintTransparency.zip(stream)
// ){
// 		try{
// 			FingerprintTemplate probe = new FingerprintTemplate(
// 				new FingerprintImage()
// 					.dpi(500)
// 					.decode(Files.readAllBytes(Paths.get("/home/vania/Downloads/gs-rest-service-master/complete/src/resources/res.jpeg"))));
// 			FingerprintTemplate candidate = new FingerprintTemplate(
// 				new FingerprintImage()
// 					.dpi(500)
// 					.decode(Files.readAllBytes(Paths.get("/home/vania/Downloads/gs-rest-service-master/complete/src/resources/res2.jpg"))));
			
// 			double score = new FingerprintMatcher()
// 				.index(probe)
// 				.match(candidate);
// 				boolean matches = score >= 40;
// 				if (matches){
// 					res = "Matched with scored " + Double. toString(score);
// 				} else {
// 				res = "Doesn't Matched with scored " + Double. toString(score);
// 				}
// 			return new UserDetails(counter.incrementAndGet(), String.format(template, res));
		
// 		} catch (Exception e ){
// 			e.printStackTrace();
// 			}
// 		}catch(Exception e) {
// 				e.printStackTrace();
// 				// res.body(e.getMessage());
// 		}
		
// 		 return new UserDetails(counter.incrementAndGet(), String.format(template, res));

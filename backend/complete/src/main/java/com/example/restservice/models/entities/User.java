package com.example.restservice.models.entities;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import javax.persistence.Entity;

@Builder
@Data
public class User {
	
   	private String name;
	
    private double score;


	public User(String content,double score) {
		this.score = score;
		this.name = content;
		
	}

    public double getScore() {
        return this.score;
    }

    public void setScore(double score) {
        this.score = score;
    }

	

	public String getName() {
		return this.name;
	}

	
	public void setName(String name) {
		this.name = name;
	}

	


}

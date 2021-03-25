package com.example.restservice.models.entities;


import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Lob;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class CandidatesCache {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
	private long id;
    private String name;
 
	@Lob
   	private byte[] template;


	public CandidatesCache(String name, byte[] template) {
		
		this.name = name;
        this.template= template;
		
	}

	public long getId() {
		return id;
	}


	public String getName() {
		return this.name;
	}

	public void setId(long id) {
		this.id = id;
	}
	public void setName(String name) {
		this.name = name;
	}

	public byte[] getTemplate() {
		return this.template;
	}

	public void setTemplate(byte[] template) {
		this.template = template;
	}

	// public byte[] getTemplate() {
	// 	return this.template;
	// }

	// public void setTemplate(byte[] template) {
	// 	this.template = template;
	// }


}

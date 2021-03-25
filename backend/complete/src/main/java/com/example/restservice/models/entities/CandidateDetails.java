package com.example.restservice.models.entities;
import com.machinezoo.sourceafis.FingerprintTemplate;

// import org.springframework.web.bind.annotation.GetMapping;
public class CandidateDetails {
	long id;
   	String name;
	
    FingerprintTemplate template;


	public CandidateDetails(long id, String content,FingerprintTemplate template) {
		this.id = id;
		this.name = content;
		this.template = template;
		
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

	public FingerprintTemplate getTemplate() {
		return this.template;
	}

	public void setTemplate(FingerprintTemplate template) {
		this.template = template;
	}

	// public FingerprintTemplate getTemplate() {
	// 	return this.template;
	// }

	// public void setTemplate(FingerprintTemplate template) {
	// 	this.template = template;
	// }


}

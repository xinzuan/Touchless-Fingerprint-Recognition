package com.example.restservice.models.outbounds.wrapper;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class DataResponse<T> extends BaseResponse {

  private T data;

  @Builder(builderMethodName = "dataBuilder")
  public DataResponse(int code, String status, T data) {
    super(code, status);
    this.data = data;
  }

}

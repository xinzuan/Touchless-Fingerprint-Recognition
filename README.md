# Touchless-Fingerprint-Recognition
Fingerprints are unique patterns of whorls and lines found on the fingertips, serving as a reliable means of individual identification. They maintain their consistency and uniqueness throughout a person's lifetime, ensuring that no two individuals, including identical twins, share the same fingerprint pattern. Traditionally, capturing fingerprints required physical contact with a reader or scanner for biometric identification purposes.

However, in the advancement of technology and the emphasis on social distancing, touchless fingerprint recognition has gained significant popularity. The challenge lies in enhancing the resolution and contrast of fingerprints without the use of sensors in touchless fingerprint systems. 

This project aims to address this challenge by exploring the application of super resolution techniques, which can effectively enhance the resolution and capture finer patterns. In this project, the fingerprint images are captured using a android phone camera.

## Prerequisite
The verification process can be done by uploading an image or directly capture the fingerprint by using a third party application. The simple UI interface:
![alt text](/assets/ui.png)


### Backend
1. Navigate to backend/complete
2. Run ``` ./mvnw spring-boot:run```

### Frontend
1. Navigate to frontend
2. Run ``` python3 main.py```

### Connect android phone camera (Direct capture)
1. Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en&pli=1) application on your mobile phone
2. Open IP Webcam application and simply locate and click the "Start server" option located at the bottom of the interface.
3. Locate the host on the bottom of the screen
4. Both the mobile phone and laptop/PC must be connected to the same network.

## Demo

## Library
1. OpenCV
2. Sckit-learn
3. Numpy
4. Pygame
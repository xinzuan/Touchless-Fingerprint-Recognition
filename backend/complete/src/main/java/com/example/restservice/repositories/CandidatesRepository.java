
package com.example.restservice.repositories;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.example.restservice.models.entities.CandidatesCache;

public interface CandidatesRepository extends JpaRepository<CandidatesCache, Long> {

// path = 'D:/OpenCV/Scripts/Images'
// cv2.imwrite(os.path.join(path , 'waka.jpg'), img)
// cv2.waitKey(0)
}
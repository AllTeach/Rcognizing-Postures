# 🧘 Pilates Posture Recognition - Student Guide

## Project Overview

You're going to build an AI model that can recognize different Pilates postures in real-time! This model will be used in an Android app to:

1. **Identify** which Pilates pose someone is doing
2. **Track** how long they hold each pose
3. **(Future)** Correct their posture with feedback

---

## 📋 Prerequisites

### What You Need:

1. **A Google Account** (for Google Colab - it's free!)
2. **Training Data**: Videos or images of Pilates poses
   - At least 2-3 different poses to start
   - 10-20 examples per pose (more is better!)
   - Clear images showing the full body
3. **No coding experience required!** - Just follow the steps

### Recommended Pilates Poses to Start With:

Choose poses that look **very different** from each other:
- ✅ **Plank** (straight body, face down)
- ✅ **Bridge** (lying on back, hips raised)
- ✅ **Hundred** (lying on back, legs and head raised)
- ✅ **Roll Up** (sitting, reaching forward)

---

## 📂 Step 1: Prepare Your Training Data

### Folder Structure:

Create a folder called `pilates_dataset` with subfolders for each pose:

```
pilates_dataset/
├── plank/
│   ├── plank1.jpg
│   ├── plank2.jpg
│   ├── plank3.mp4
│   └── ... (more images/videos)
├── bridge/
│   ├── bridge1.jpg
│   ├── bridge2.jpg
│   └── ... (more images/videos)
├── hundred/
│   ├── hundred1.jpg
│   └── ... (more images/videos)
└── roll_up/
    ├── rollup1.jpg
    └── ... (more images/videos)
```

### Tips for Good Training Data:

✅ **DO:**
- Use clear, well-lit photos/videos
- Show the full body in frame
- Include different people if possible
- Vary the camera angles slightly
- Ensure the pose is performed correctly

❌ **DON'T:**
- Use blurry or dark images
- Cut off parts of the body
- Mix different poses in the same folder
- Use images where the person is too far away

### Where to Get Training Data:

1. **Record yourself** doing each pose (easiest!)
2. **Ask friends** to demonstrate
3. **Extract frames** from YouTube Pilates videos (for learning purposes)
4. **Use stock photos** from free sites like Pexels or Unsplash

---

## 💻 Step 2: Open Google Colab

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with your Google Account
3. Click **File → Upload notebook**
4. Upload the `Pilates_Posture_Recognition_Training.ipynb` file

---

## 📤 Step 3: Upload Your Training Data

### Option A: Direct Upload (Simple)

1. Click the **folder icon** 📁 on the left sidebar
2. Drag and drop your entire `pilates_dataset` folder
3. Wait for upload to complete

⚠️ **Note:** Files uploaded this way are deleted when you close Colab!

### Option B: Google Drive (Recommended)

1. Upload `pilates_dataset` to your Google Drive
2. In the notebook, run the cell that says:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update the path to point to your Drive folder

---

## ▶️ Step 4: Run the Notebook

### How to Run Cells:

1. Click on a cell (the gray box with code)
2. Press the **Play button** ▶️ on the left of the cell
3. Wait for it to finish (you'll see a green checkmark ✓)
4. Move to the next cell

### Run in Order:

Run **EVERY cell from top to bottom**. Don't skip any!

### What Each Section Does:

| Section | What It Does | Time |
|---------|--------------|------|
| Step 1 | Installs required libraries | 1-2 min |
| Step 2 | Loads the libraries | 10 sec |
| Step 3 | Sets up MediaPipe (pose detection) | 5 sec |
| Step 4 | Creates angle calculation function | 5 sec |
| Step 5 | Creates feature extraction function | 5 sec |
| Step 6 | Sets the path to your data | 5 sec |
| Step 7 | **Processes all your images/videos** | 2-5 min |
| Step 8 | Prepares data for training | 10 sec |
| Step 9 | **Trains Random Forest model** | 30 sec |
| Step 10 | (Optional) Trains Neural Network | 1-2 min |
| Step 11 | **Converts to Android format** | 30 sec |
| Step 12 | Saves pose labels | 5 sec |
| Step 13 | **Downloads your model** | 10 sec |

**Total Time: ~10-15 minutes**

---

## 📊 Step 5: Check Your Results

### After Step 9 (Random Forest Training):

Look for these numbers:

```
Training Accuracy: 95.00%
Testing Accuracy: 90.00%
```

**What's Good?**
- ✅ Testing accuracy **above 80%** = Good!
- ✅ Testing accuracy **above 90%** = Excellent!
- ⚠️ Testing accuracy **below 70%** = Need more/better data

**If accuracy is low:**
1. Collect more training images (aim for 30+ per pose)
2. Make sure poses are performed correctly
3. Check that images are clear and well-lit
4. Try training the Neural Network (Step 10)

### Feature Importance:

You'll see which body angles matter most:

```
Most Important Angles for Classification:
1. Left Hip: 0.215
2. Right Knee: 0.198
3. Left Knee: 0.187
...
```

This tells you which joints are most important for distinguishing poses!

---

## 💾 Step 6: Download Your Model

### Files You Need:

After running Step 13, download these files:

1. **`pilates_model.tflite`** - Your trained AI model
2. **`labels.txt`** - List of pose names

### How to Download:

**Method 1:** Run the download cell (Step 13, last cell)
- Files will download to your browser's download folder

**Method 2:** Manual download
1. Click folder icon 📁 on the left
2. Right-click each file
3. Select "Download"

---

## 📱 Step 7: Use in Your Android App

### Add Files to Android Project:

1. In Android Studio, navigate to: `app/src/main/assets/`
2. Copy both files there:
   - `pilates_model.tflite`
   - `labels.txt`

### Add Dependencies:

In your `app/build.gradle`:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'com.google.mediapipe:tasks-vision:0.10.0'
    implementation 'androidx.camera:camera-core:1.3.0'
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
}
```

### Basic Android Implementation:

```java
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.PointF;
import org.tensorflow.lite.Interpreter;
import java.io.*;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

public class PilatesDetector {
    
    private Interpreter interpreter;
    private List<String> labels;
    
    // Pose tracking
    private String currentPose = "";
    private long poseStartTime = 0L;
    
    public PilatesDetector(Context context) {
        try {
            // Load model
            MappedByteBuffer modelFile = loadModelFile(context, "pilates_model.tflite");
            interpreter = new Interpreter(modelFile);
            
            // Load labels
            InputStream inputStream = context.getAssets().open("labels.txt");
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            labels = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                labels.add(line);
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public PoseResult detectPose(List<Landmark> poseLandmarks) {
        // 1. Calculate angles from landmarks
        float[] angles = calculateAngles(poseLandmarks);
        
        // 2. Run inference
        float[][] inputArray = new float[1][angles.length];
        inputArray[0] = angles;
        float[][] outputArray = new float[1][labels.size()];
        interpreter.run(inputArray, outputArray);
        
        // 3. Get prediction
        float[] predictions = outputArray[0];
        int maxIndex = 0;
        float maxValue = predictions[0];
        for (int i = 1; i < predictions.length; i++) {
            if (predictions[i] > maxValue) {
                maxValue = predictions[i];
                maxIndex = i;
            }
        }
        float confidence = predictions[maxIndex];
        String predictedPose = labels.get(maxIndex);
        
        // 4. Track duration
        long duration;
        if (predictedPose.equals(currentPose)) {
            duration = (System.currentTimeMillis() - poseStartTime) / 1000;
        } else {
            currentPose = predictedPose;
            poseStartTime = System.currentTimeMillis();
            duration = 0;
        }
        
        return new PoseResult(predictedPose, confidence, duration);
    }
    
    private float[] calculateAngles(List<Landmark> landmarks) {
        // Same angle calculations as in Python notebook
        // See detailed implementation in notebook
        
        return new float[] {
            calculateAngle(landmarks.get(11), landmarks.get(13), landmarks.get(15)), // Left elbow
            calculateAngle(landmarks.get(12), landmarks.get(14), landmarks.get(16)), // Right elbow
            calculateAngle(landmarks.get(13), landmarks.get(11), landmarks.get(23)), // Left shoulder
            calculateAngle(landmarks.get(14), landmarks.get(12), landmarks.get(24)), // Right shoulder
            calculateAngle(landmarks.get(11), landmarks.get(23), landmarks.get(25)), // Left hip
            calculateAngle(landmarks.get(12), landmarks.get(24), landmarks.get(26)), // Right hip
            calculateAngle(landmarks.get(23), landmarks.get(25), landmarks.get(27)), // Left knee
            calculateAngle(landmarks.get(24), landmarks.get(26), landmarks.get(28)), // Right knee
            calculateAngle(landmarks.get(23), landmarks.get(11), landmarks.get(12)), // Left torso
            calculateAngle(landmarks.get(24), landmarks.get(12), landmarks.get(11))  // Right torso
        };
    }
    
    private float calculateAngle(Landmark a, Landmark b, Landmark c) {
        float baX = a.getX() - b.getX();
        float baY = a.getY() - b.getY();
        float bcX = c.getX() - b.getX();
        float bcY = c.getY() - b.getY();
        
        float dot = baX * bcX + baY * bcY;
        float magnitudeBA = (float) Math.sqrt(baX * baX + baY * baY);
        float magnitudeBC = (float) Math.sqrt(bcX * bcX + bcY * bcY);
        
        float cosine = dot / (magnitudeBA * magnitudeBC);
        cosine = Math.max(-1f, Math.min(1f, cosine)); // Clamp to [-1, 1]
        float angle = (float) Math.acos(cosine);
        
        return (float) Math.toDegrees(angle);
    }
    
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}

class PoseResult {
    private String poseName;
    private float confidence;
    private long duration;
    
    public PoseResult(String poseName, float confidence, long duration) {
        this.poseName = poseName;
        this.confidence = confidence;
        this.duration = duration;
    }
    
    public String getPoseName() { return poseName; }
    public float getConfidence() { return confidence; }
    public long getDuration() { return duration; }
}
```

---

## 🎯 Common Issues & Solutions

### Issue 1: "No data collected"

**Cause:** Dataset folder not found or empty

**Solution:**
- Check folder name is exactly `pilates_dataset`
- Check folder structure (subfolders for each pose)
- Make sure upload completed

### Issue 2: "Low accuracy (below 70%)"

**Causes:**
- Not enough training data
- Poses too similar
- Poor quality images

**Solutions:**
- Collect 30-50 samples per pose
- Choose more distinct poses
- Use clearer, better-lit images
- Try the Neural Network (Step 10)

### Issue 3: "Error extracting features"

**Cause:** MediaPipe can't detect body in image

**Solutions:**
- Ensure full body is visible
- Improve lighting
- Remove images where person is too far/too close
- Check image quality

### Issue 4: Model file too large

**Solution:**
- The model should be < 1 MB
- If larger, use more quantization:
  ```python
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  ```

---

## 🚀 Next Steps: Adding Posture Correction

Once basic recognition works, you can add posture correction:

### Phase 2 Features:

1. **Define ideal angles** for each pose
2. **Compare** detected angles to ideal
3. **Provide feedback**:
   - "Bend your left knee more (currently 145°, should be 90°)"
   - "Straighten your back"
   - "Lower your hips"

### Example Code:

```java
// Define ideal angles for each pose
Map<String, Map<String, Float>> idealAngles = new HashMap<>();

Map<String, Float> plankAngles = new HashMap<>();
plankAngles.put("left_elbow", 180f);
plankAngles.put("right_elbow", 180f);
plankAngles.put("left_hip", 180f);
plankAngles.put("right_hip", 180f);
idealAngles.put("plank", plankAngles);

Map<String, Float> bridgeAngles = new HashMap<>();
bridgeAngles.put("left_knee", 90f);
bridgeAngles.put("right_knee", 90f);
bridgeAngles.put("left_hip", 135f);
bridgeAngles.put("right_hip", 135f);
idealAngles.put("bridge", bridgeAngles);

public List<String> provideFeedback(String pose, float[] detectedAngles) {
    List<String> feedback = new ArrayList<>();
    Map<String, Float> ideal = idealAngles.get(pose);
    if (ideal == null) return feedback;
    
    String[] angleNames = {
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_torso", "right_torso"
    };
    
    for (int index = 0; index < detectedAngles.length; index++) {
        String angleName = angleNames[index];
        Float idealAngle = ideal.get(angleName);
        
        if (idealAngle != null) {
            float difference = Math.abs(detectedAngles[index] - idealAngle);
            
            if (difference > 15) {  // Tolerance: 15 degrees
                String direction = detectedAngles[index] > idealAngle ? "less" : "more";
                feedback.add("Bend your " + angleName.replace('_', ' ') + " " + direction);
            }
        }
    }
    
    return feedback;
}
```

---

## 📚 Learning Resources

### Understanding the Concepts:

1. **Pose Estimation:**
   - [MediaPipe Pose Overview](https://google.github.io/mediapipe/solutions/pose.html)
   - How computers detect body keypoints

2. **Machine Learning Basics:**
   - [What is a Random Forest?](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
   - [Neural Networks Explained](https://www.youtube.com/watch?v=aircAruvnKk)

3. **TensorFlow Lite:**
   - [TFLite for Android](https://www.tensorflow.org/lite/android)
   - Converting models for mobile

### Practice Projects:

1. Start with 2 very different poses (e.g., plank vs. bridge)
2. Add a 3rd pose once the first two work well
3. Gradually build up to 5-10 poses
4. Experiment with adding posture correction

---

## ✅ Checklist: Before You Start

- [ ] Google account ready
- [ ] Training images/videos collected (10+ per pose)
- [ ] Images organized in correct folder structure
- [ ] `pilates_dataset` folder created
- [ ] Notebook file downloaded
- [ ] Read through this guide once

## ✅ Checklist: After Training

- [ ] Training completed successfully
- [ ] Accuracy is acceptable (>80%)
- [ ] `pilates_model.tflite` downloaded
- [ ] `labels.txt` downloaded
- [ ] Files added to Android project
- [ ] Dependencies added to build.gradle
- [ ] Ready to implement in app!

---

## 💬 Need Help?

### Common Questions:

**Q: How many images do I need?**
A: Minimum 10-20 per pose, but 30-50 is better. More data = better accuracy!

**Q: Can I use videos?**
A: Yes! The notebook extracts frames from videos automatically.

**Q: Which model should I use - Random Forest or Neural Network?**
A: Start with Random Forest (simpler). If accuracy is low, try Neural Network.

**Q: How long does training take?**
A: 10-15 minutes for the entire notebook with ~100 images.

**Q: Do I need a powerful computer?**
A: No! Google Colab provides free GPU access. Just need internet.

**Q: Can I retrain with more data later?**
A: Yes! Just add more images to your dataset and run the notebook again.

**Q: What if two poses look similar?**
A: Collect more examples and ensure they're performed correctly. Or choose more distinct poses.

---

## 🎉 You're Ready!

Follow the steps, run the notebook, and you'll have a working Pilates pose recognition model!

**Remember:**
- Take it one step at a time
- Read the comments in the code
- Don't skip any cells
- Check your results after each major step

Good luck! 🧘‍♀️💪
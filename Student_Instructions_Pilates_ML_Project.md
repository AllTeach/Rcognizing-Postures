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

```kotlin
class PilatesDetector(context: Context) {
    
    private val interpreter: Interpreter
    private val labels: List<String>
    
    // Pose tracking
    private var currentPose = ""
    private var poseStartTime = 0L
    
    init {
        // Load model
        val modelFile = loadModelFile(context, "pilates_model.tflite")
        interpreter = Interpreter(modelFile)
        
        // Load labels
        labels = context.assets.open("labels.txt")
            .bufferedReader()
            .readLines()
    }
    
    fun detectPose(poseLandmarks: List<Landmark>): PoseResult {
        // 1. Calculate angles from landmarks
        val angles = calculateAngles(poseLandmarks)
        
        // 2. Run inference
        val inputArray = arrayOf(angles)
        val outputArray = Array(1) { FloatArray(labels.size) }
        interpreter.run(inputArray, outputArray)
        
        // 3. Get prediction
        val predictions = outputArray[0]
        val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: 0
        val confidence = predictions[maxIndex]
        val predictedPose = labels[maxIndex]
        
        // 4. Track duration
        val duration = if (predictedPose == currentPose) {
            (System.currentTimeMillis() - poseStartTime) / 1000
        } else {
            currentPose = predictedPose
            poseStartTime = System.currentTimeMillis()
            0
        }
        
        return PoseResult(
            poseName = predictedPose,
            confidence = confidence,
            duration = duration
        )
    }
    
    private fun calculateAngles(landmarks: List<Landmark>): FloatArray {
        // Same angle calculations as in Python notebook
        // See detailed implementation in notebook
        
        return floatArrayOf(
            calculateAngle(landmarks[11], landmarks[13], landmarks[15]), // Left elbow
            calculateAngle(landmarks[12], landmarks[14], landmarks[16]), // Right elbow
            calculateAngle(landmarks[13], landmarks[11], landmarks[23]), // Left shoulder
            calculateAngle(landmarks[14], landmarks[12], landmarks[24]), // Right shoulder
            calculateAngle(landmarks[11], landmarks[23], landmarks[25]), // Left hip
            calculateAngle(landmarks[12], landmarks[24], landmarks[26]), // Right hip
            calculateAngle(landmarks[23], landmarks[25], landmarks[27]), // Left knee
            calculateAngle(landmarks[24], landmarks[26], landmarks[28]), // Right knee
            calculateAngle(landmarks[23], landmarks[11], landmarks[12]), // Left torso
            calculateAngle(landmarks[24], landmarks[12], landmarks[11])  // Right torso
        )
    }
    
    private fun calculateAngle(a: Landmark, b: Landmark, c: Landmark): Float {
        val ba = PointF(a.x - b.x, a.y - b.y)
        val bc = PointF(c.x - b.x, c.y - b.y)
        
        val dot = ba.x * bc.x + ba.y * bc.y
        val magnitudeBA = sqrt(ba.x * ba.x + ba.y * ba.y)
        val magnitudeBC = sqrt(bc.x * bc.x + bc.y * bc.y)
        
        val cosine = dot / (magnitudeBA * magnitudeBC)
        val angle = acos(cosine.coerceIn(-1f, 1f))
        
        return Math.toDegrees(angle.toDouble()).toFloat()
    }
}

data class PoseResult(
    val poseName: String,
    val confidence: Float,
    val duration: Long
)
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

```kotlin
// Define ideal angles for each pose
val idealAngles = mapOf(
    "plank" to mapOf(
        "left_elbow" to 180f,
        "right_elbow" to 180f,
        "left_hip" to 180f,
        "right_hip" to 180f
    ),
    "bridge" to mapOf(
        "left_knee" to 90f,
        "right_knee" to 90f,
        "left_hip" to 135f,
        "right_hip" to 135f
    )
)

fun provideFeedback(pose: String, detectedAngles: FloatArray): List<String> {
    val feedback = mutableListOf<String>()
    val ideal = idealAngles[pose] ?: return feedback
    
    val angleNames = listOf("left_elbow", "right_elbow", "left_shoulder", 
                           "right_shoulder", "left_hip", "right_hip",
                           "left_knee", "right_knee", "left_torso", "right_torso")
    
    detectedAngles.forEachIndexed { index, angle ->
        val angleName = angleNames[index]
        val idealAngle = ideal[angleName]
        
        if (idealAngle != null) {
            val difference = abs(angle - idealAngle)
            
            if (difference > 15) {  // Tolerance: 15 degrees
                val direction = if (angle > idealAngle) "less" else "more"
                feedback.add("Bend your ${angleName.replace('_', ' ')} $direction")
            }
        }
    }
    
    return feedback
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
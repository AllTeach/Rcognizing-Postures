# Student Instructions for Pilates ML Project

Content from lines 1-234 as it is...


// Java implementation of PilatesDetector class shown below

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


// Content from lines 432-518 as it is...
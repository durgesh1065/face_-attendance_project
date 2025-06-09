1.Algorithm Selection
  Uses face recognition based on HOG (Histogram of Oriented Gradients) or CNN model to encode facial features.
  Compares known encodings to detected faces using Euclidean distance.
2.Input Data
  Captured a real-time face image
  Pre-saved encodings of enrolled students
3.Training Process
  No traditional ML training needed
  Encodings are directly stored and used for matching
4.Prediction & Logging
  The detected face is compared with saved encodings
  If match found → Display name & roll number → Save to CSV
5.Deployment
  Flask-based lightweight web app
  Easy to operate: Touch-free, real-time, fast
  Can be deployed in schools, colleges, or offices


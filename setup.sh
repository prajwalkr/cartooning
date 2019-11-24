pip -r requirements.txt
mkdir -p results
mkdir -p data
cd data
wget https://github.com/codeniko/shape_predictor_81_face_landmarks/blob/master/shape_predictor_81_face_landmarks.dat
cd ..
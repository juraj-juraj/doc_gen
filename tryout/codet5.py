import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Model:
    def __init__(self, device: str) -> None:
        """Initialize model and tokenizer
        This works on computer DELL G5 5590
        Model summary:
                codet5p-770m and codet5p-220m works well, generating code
                codet5-small can be run, but not generating anything usable
                codet5p-2b doesnt run, due to errors I cannot solve, seems it is too performance demanding
        """
        self._device = device

        checkpoint = "juraj-juraj/docstring-t5"  # "Salesforce/codet5p-770m"
        logging.debug(f"Loading checkpoint {checkpoint}...")
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        logging.info("Tokenizer loaded.")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self._device)
        logging.info("Model loaded.")

    def generate(self, prompt: str, max_length: int = 150) -> str:
        the_prompt = """Generate docstring to python function: 
    def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf"""

        inputs = self._tokenizer(the_prompt, return_tensors="pt").to(self._device)
        output_sequences = self._model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
        )
        return self._tokenizer.decode(output_sequences[0], skip_special_tokens=True)

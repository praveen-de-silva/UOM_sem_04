/*
  Timed Laboratory Exercise #1 - Task 1
  8-Key Tone Generator
  Name: De Silva B. K. P.
  Index Number: 230123K
*/

// Pin assignments
const int buttonPins[8] = {2, 3, 4, 5, 6, 7, 8, 9};
const int speakerPin = 11;

// Frequencies for each key
const int frequencies[8] = {300, 400, 500, 600, 700, 800, 900, 1000};

void setup() {
  for (int i = 0; i < 8; i++) {
    pinMode(buttonPins[i], INPUT); // Using external pull-down resistors
  }
  pinMode(speakerPin, OUTPUT);
}

void loop() {
  int totalFreq = 0;
  bool anyPressed = false;

  // Sum up frequencies of all pressed keys
  for (int i = 0; i < 8; i++) {
    if (digitalRead(buttonPins[i]) == HIGH) {
      totalFreq += frequencies[i];
      anyPressed = true;
    }
  }

  if (anyPressed) {
    // Check if keys are still held after 20ms
    tone(speakerPin, totalFreq);
    delay(20);

    // Re-check if any key is still held
    bool stillHeld = false;
    for (int i = 0; i < 8; i++) {
      if (digitalRead(buttonPins[i]) == HIGH) {
        stillHeld = true;
        break;
      }
    }

    // If released after 20ms, stop tone; if held, keep playing
    if (!stillHeld) {
      noTone(speakerPin);
    }
    // If still held, loop continues and tone keeps playing
  } else {
    noTone(speakerPin);
  }
}
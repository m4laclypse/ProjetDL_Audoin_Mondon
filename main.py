from flappyBird import FlappyBird

if __file__== "__main__":
    flappy = FlappyBird()
    while True:
        state = flappy.getState()
        score = flappy.getScore()

        entry = fonctionTropIntelligente(state)
        flappy.nextFrame(manual=True, entry=entry)

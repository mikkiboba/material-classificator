import time
import training

def main() -> None:
    training.execute_model_complete(
        learning_rate=0.0001,
        batch_size=64,
        n_epochs=120,
        load_dataset=False,
        fft=False
    )

if __name__ == "__main__":
    now = time.time()
    main()
    print(f'-- EXECUTION TIME: {time.time()-now} seconds')

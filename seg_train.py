# from vajra import Vajra

# # Instance segmentation training script
# # NOTE: Windows users running as a script should keep the main guard.

# def main():
#     model = Vajra("vajra-v1-nano-seg")  # baseline segmentation model

#     train_results = model.train(
#         data=r"F:\material\zanbiya_datasets\data.yaml",
#         epochs=300,
#         img_size=640,
#         rect=False,
#         workers=0,
#         device=0,
#         batch=32,
#         project="runs2/zanbiya_data baseline",
#         name="Vajra-baseline-2",



#     )

#     # Validate using the training dataset settings
#     metrics = model.val()
#     return train_results, metrics


# if __name__ == "__main__":
#     main()
from vajra import Vajra

# Instance segmentation training script
# NOTE: Windows users running as a script should keep the main guard.

def main():
    model = Vajra("vajra-v1-nano-seg")  # baseline segmentation model

    train_results = model.train(
        data=r"C:\Users\Administrator\Desktop\datasets\data.yaml",
        epochs=200,
        img_size=640,
        # rect=True,
        workers=0,
        device=0,
        batch=32,
        project="keshihua",
        name="our",
        seed=42,


    )

    # Validate using the training dataset settings
    metrics = model.val()
    return train_results, metrics


if __name__ == "__main__":
    main()

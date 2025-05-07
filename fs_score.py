from fsscore.score import Scorer
from fsscore.models.ranknet import LitRankNet


def main():
    print("Hello from geoldm-moo!")
    # 1) load pre-trained model or choose path to own model
    model = LitRankNet.load_from_checkpoint(
        "./fsscore_models/pretrain_graph_GGLGGL_ep242_best_valloss.ckpt"
    )

    # 2) initialize scorer
    scorer = Scorer(model=model, device="cpu")
    smiles = ["C1=CC=C2C(=C1)C=CC3=C2C=CC=C3C4=CC=CC=C4"]
    # 3) predict scores given a list of SMILES
    scores = scorer.score(smiles)
    print(scores)


if __name__ == "__main__":
    main()

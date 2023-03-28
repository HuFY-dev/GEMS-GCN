import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader
import argparse


# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, emb_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hid_feats)
        self.conv2 = GraphConv(hid_feats, emb_feats)
        self.dropout = nn.Dropout(0.2)

    def forward(self, graph, x):
        # Move the input to the device of the graph
        x = x.to(graph.device)

        # Compute node embeddings
        x = self.conv1(graph, x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(graph, x)
        return x


# Define the link prediction model
class LinkPredictor(nn.Module):
    def __init__(self, emb_feats, hid_feats, out_feats):
        super(LinkPredictor, self).__init__()
        self.lin1 = nn.Linear(emb_feats, hid_feats)
        self.lin2 = nn.Linear(hid_feats, out_feats)
        self.dropout = nn.Dropout(0.2)

    def forward(self, i_emb, j_emb):
        # Compute the link prediction scores
        x = i_emb * j_emb
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        # Use sigmoid to normalize the scores to be between 0 and 1
        return torch.sigmoid(x)


def train(model, predictor, graph, split_edge, optimizer, batch_size):
    # Set the model to train mode
    model.train()
    predictor.train()

    # Find the device for the graph
    device = graph.device

    # Get the train edges and move them to the device
    train_edges = split_edge["train"]["edge"].to(device)
    train_feats = graph.ndata["feat"]

    # Initialize the loss and number of examples
    total_loss = total_examples = 0

    # Sample a batch of edges
    for perm in DataLoader(range(train_edges.size(0)), batch_size, shuffle=True):
        # Clear the gradients
        optimizer.zero_grad()

        # Compute node embeddings
        emb_x = model(graph, train_feats)

        # Predict positive edges
        edge = train_edges[perm].t().to(device)
        pos_out = predictor(emb_x[edge[0]], emb_x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Generate negative edges
        edge = torch.randint(
            0, graph.num_nodes(), edge.size(), dtype=torch.long, device=emb_x.device
        )

        # Predict negative edges
        neg_out = predictor(emb_x[edge[0]], emb_x[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss

        # Backward and update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        # Compute the loss and number of examples
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def eval(model, predictor, graph, split_edge, evaluator, batch_size):
    # Set the model to evaluation mode
    model.eval()
    predictor.eval()

    # Find the device for the graph
    device = graph.device

    # Get different splitted edges and move them to the device
    pos_train_edges = split_edge["train"]["edge"].to(device)
    pos_valid_edges = split_edge["valid"]["edge"].to(device)
    neg_valid_edges = split_edge["valid"]["edge_neg"].to(device)
    pos_test_edges = split_edge["test"]["edge"].to(device)
    neg_test_edges = split_edge["test"]["edge_neg"].to(device)
    feats = graph.ndata["feat"]

    # Compute node embeddings
    emb_x = model(graph, feats)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edges.size(0)), batch_size, shuffle=False):
        edge = pos_train_edges[perm].t()
        pos_train_preds.append(
            predictor(emb_x[edge[0]], emb_x[edge[1]]).squeeze()
        )
    pos_train_preds = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edges.size(0)), batch_size, shuffle=False):
        edge = pos_valid_edges[perm].t()
        pos_valid_preds.append(
            predictor(emb_x[edge[0]], emb_x[edge[1]]).squeeze()
        )
    pos_valid_preds = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edges.size(0)), batch_size, shuffle=False):
        edge = neg_valid_edges[perm].t()
        neg_valid_preds.append(
            predictor(emb_x[edge[0]], emb_x[edge[1]]).squeeze().cpu()
        )
    neg_valid_preds = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edges.size(0)), batch_size, shuffle=False):
        edge = pos_test_edges[perm].t()
        pos_test_preds.append(predictor(emb_x[edge[0]], emb_x[edge[1]]).squeeze().cpu())
    pos_test_preds = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edges.size(0)), batch_size, shuffle=False):
        edge = neg_test_edges[perm].t()
        neg_test_preds.append(predictor(emb_x[edge[0]], emb_x[edge[1]]).squeeze().cpu())
    neg_test_preds = torch.cat(neg_test_preds, dim=0)

    evaluator.K = 50
    train_hits = evaluator.eval(
        {
            "y_pred_pos": pos_train_preds,
            "y_pred_neg": neg_valid_preds,
        }
    )["hits@50"]
    valid_hits = evaluator.eval(
        {
            "y_pred_pos": pos_valid_preds,
            "y_pred_neg": neg_valid_preds,
        }
    )["hits@50"]
    test_hits = evaluator.eval(
        {
            "y_pred_pos": pos_test_preds,
            "y_pred_neg": neg_test_preds,
        }
    )["hits@50"]

    results = [train_hits, valid_hits, test_hits]

    return results


def main():
    # Create the argument parser and parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=int,
        default=100,
        help="specify how many epochs to run before saving the model",
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="specify how many epochs to run in total"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64 * 1024,
        help="specify the batch size during training and testing",
    )
    parser.add_argument("--eval_steps", type=int, default=1)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    dataset = DglLinkPropPredDataset(name="ogbl-collab")
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    # Add self-loops to the graph
    graph = graph.add_self_loop()

    # Prepare features
    in_feats = graph.ndata["feat"].shape[1]
    train_feats = graph.ndata["feat"]

    # Create model and optimizer
    emb_feats = 256
    hid_feats = 256
    model = GCN(in_feats, hid_feats,emb_feats).to(device)
    predictor = LinkPredictor(emb_feats, hid_feats, 1).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=0.0005
    )

    # Move graph to device
    graph = graph.to(device)

    # Set the number of epochs and the batch size
    batch_size = args.batch_size
    epochs = args.epochs + 1
    for epoch in range(epochs):
        print(f"Training epoch {epoch}...")

        # Train model
        loss = train(model, predictor, graph, split_edge, optimizer, batch_size)
        print(f"train loss: {loss}")

        # Evaluate model
        if epoch % args.eval_steps == 0:
            evaluator = Evaluator(name="ogbl-collab")
            result = eval(model, predictor, graph, split_edge, evaluator, batch_size)
            print(f"hits@50: {result}")

        print(f"Finished epoch {epoch}\n")

        # Save the model checkpoint if epoch is multiple of 10
        save = args.save
        if epoch % save != 0:
            continue
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            f"GCN_checkpoint_epoch{epoch}.pt",
        )


if __name__ == "__main__":
    main()

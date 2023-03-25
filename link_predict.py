import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from ogb.linkproppred import DglLinkPropPredDataset
from torch.utils.data import DataLoader


# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, emb_feats):
        super(GCN, self).__init__()
        self.conv = GraphConv(in_feats, emb_feats)

    def forward(self, graph, x):
        # Set the device for the graph and input tensors
        device = graph.device

        # Move the input tensors to the device
        x = x.to(device)

        # Compute node embeddings
        x = self.conv(graph, x)
        x = F.relu(x)
        return x


# Define the link prediction model
class LinkPredictor(nn.Module):
    def __init__(self, emb_feats, out_feats):
        super(LinkPredictor, self).__init__()
        self.lin = nn.Linear(emb_feats, out_feats)

    def forward(self, i_emb, j_emb):
        # Compute the link prediction scores
        x = i_emb * j_emb
        x = self.lin(x)

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
        edge = train_edges[perm].t()
        pos_out = predictor(emb_x[edge[0]], emb_x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # # Generate negative edges
        # edge = torch.randint(
        #     0, graph.num_nodes(), edge.size(), dtype=torch.long, device=emb_x.device
        # )

        # # Predict negative edges
        # neg_out = predictor(emb_x[edge[0]], emb_x[edge[1]])
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        # loss = pos_loss + neg_loss

        # train without negative edges
        loss = pos_loss

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


def valid(model, predictor, graph, split_edge):
    # Set the model to evaluation mode
    model.eval()
    predictor.eval()

    # Find the device for the graph
    device = graph.device

    # Get the valid edges and move them to the device
    valid_edges = split_edge["valid"]["edge"].to(device)
    valid_feats = graph.ndata["feat"]

    # Compute node embeddings
    emb_x = model(graph, valid_feats)

    # Predict edges
    edge = valid_edges.t()
    out = predictor(emb_x[edge[0]], emb_x[edge[1]])
    loss = -torch.log(out + 1e-15).mean()

    return loss.item()


def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    dataset = DglLinkPropPredDataset(name="ogbl-collab")
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    # Prepare features
    in_feats = graph.ndata["feat"].shape[1]
    train_feats = graph.ndata["feat"]

    # Create model and optimizer
    emb_feats = 128
    model = GCN(in_feats, emb_feats).to(device)
    predictor = LinkPredictor(emb_feats, 1).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=0.0005
    )

    # Move graph to device
    graph = graph.to(device)

    # Set the number of epochs and the batch size
    batch_size = 4096
    epochs = 10
    for epoch in range(epochs):
        print(f"Training epoch {epoch}...")

        # Train model
        loss = train(model, predictor, graph, split_edge, optimizer, batch_size)
        print(f"Epoch {epoch}, train loss: {loss}")

        # Validate model
        loss = valid(model, predictor, graph, split_edge)
        print(f"Epoch {epoch}, valid loss: {loss}")

        print(f"Finished epoch {epoch}\n")

        # Save the model checkpoint if epoch is multiple of 10
        if epoch % 10 != 0:
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

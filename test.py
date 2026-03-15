if True:
      import sys, torch
      sys.path.insert(0, '.')
      from torch_graph import capture, GraphVisualizer
      import torch.nn as nn

      class MLP(nn.Module):
          def __init__(self):
              super().__init__()
              self.fc1 = nn.Linear(8, 16)
              self.fc2 = nn.Linear(16, 4)
          def forward(self, x):
              x = torch.relu(self.fc1(x))
              return self.fc2(x)

      mlp = MLP()
      result, gc = capture.capture_graphs(mlp, torch.randn(2, 8))
      graphs = list(gc)
      viz = GraphVisualizer(graphs[0])
      viz.save_html('outputs/mlp_forward_v4.html', title='MLP Forward')
      print('Saved: outputs/mlp_forward_v4.html')

      sys.path.insert(0, 'test_repo')
      from model import NanoGPT
      gpt = NanoGPT()
      result2, gc2 = capture.capture_graphs(gpt, torch.randint(0, 64, (2, 16)))
      graphs2 = list(gc2)
      viz2 = GraphVisualizer(graphs2[0])
      viz2.save_html('outputs/nanogpt_forward_v4.html', title='NanoGPT Forward')
      print('Saved: outputs/nanogpt_forward_v4.html')

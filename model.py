import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
import math

class GC_Qzz_net(nn.Module):
    def __init__(self, d_model, hidden_dim, double_q = False):
        super(GC_Qzz_net, self).__init__()

        # Q1 Network
        self.fc1 = nn.Linear(3*d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Q2 Network
        self.fc4 = nn.Linear(3*d_model, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

        self.double_q = double_q

    def forward(self, z1, z2, z_g):
       # encoded_state: (batch_size, d_model)
       # action: (batch_size, action_dim)
       zz = torch.cat([z1, z2, z_g], dim=1)

       q1 = F.relu(self.fc1(zz))
       q1 = F.relu(self.fc2(q1))
       q1 = self.fc3(q1)

       if not self.double_q:
        return q1
       else:
        q2 = F.relu(self.fc4(zz))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

       return q1, q2

class GC_Vz_net(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(GC_Vz_net, self).__init__()
        self.fc1 = nn.Linear(2*d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, z1, z_g):
        zz = torch.cat([z1, z_g], dim=1)
        v = F.relu(self.fc1(zz))
        v = F.relu(self.fc2(v))
        v = self.fc3(v)
        return v

class inv_dynamics_net(nn.Module):
    def __init__(self, d_model, hidden_dim, action_dim):
        super(inv_dynamics_net, self).__init__()
        self.fc1 = nn.Linear(3*d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, z, z_p, z_g):
        zz = torch.cat([z, z_p, z_g], dim=1)
        z_next = F.relu(self.fc1(zz))
        z_next = F.relu(self.fc2(z_next))
        action = self.fc3(z_next)
        action = torch.tanh(action)
        return action

class RoPE(nn.Module):
    """Rotary Position Embedding for Mamba - 최적화된 버전"""
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 더 안정적인 주파수 계산 (오버플로우 방지)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 사전 계산된 위치 임베딩 (메모리 vs 계산 트레이드오프)
        self.register_buffer('cos_cache', None)
        self.register_buffer('sin_cache', None)
        self.cache_size = 0
        
    def _get_rope_cache(self, seq_len):
        """사전 계산된 RoPE 캐시 반환 (메모리 효율성)"""
        if self.cos_cache is None or self.cache_size < seq_len:
            # 캐시 크기 확장
            max_len = max(seq_len, self.cache_size * 2 if self.cache_size > 0 else seq_len)
            max_len = min(max_len, self.max_seq_len)
            
            t = torch.arange(max_len, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            
            # cos, sin 값을 한 번에 계산
            cos = freqs.cos()
            sin = freqs.sin()
            
            # 캐시에 저장
            self.register_buffer('cos_cache', cos)
            self.register_buffer('sin_cache', sin)
            self.cache_size = max_len
            
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]
    
    def forward(self, x, seq_len=None):
        """
        최적화된 RoPE forward pass
        
        Args:
            x: (batch_size, seq_len, d_model) 또는 (seq_len, d_model)
            seq_len: 시퀀스 길이 (None이면 x.shape[1] 사용)
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # 캐시에서 cos, sin 값 가져오기
        cos, sin = self._get_rope_cache(seq_len)
        
        # 입력 형태 처리
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, d_model)
            single_seq = True
        else:
            single_seq = False
        
        batch_size, seq_len, d_model = x.shape
        
        # d_model이 짝수인지 확인
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        # 효율적인 회전 적용
        # x를 (batch_size, seq_len, d_model//2, 2)로 reshape
        x_reshaped = x.view(batch_size, seq_len, d_model // 2, 2)
        
        # cos, sin을 적절한 차원으로 확장
        cos_expanded = cos.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, d_model//2, 1)
        sin_expanded = sin.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, d_model//2, 1)
        
        # 회전 적용: [cos(θ) -sin(θ)] [x1] = [x1*cos(θ) - x2*sin(θ)]
        #            [sin(θ)  cos(θ)] [x2]   [x1*sin(θ) + x2*cos(θ)]
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        
        # 효율적인 회전 계산
        x_rotated = torch.stack([
            x1 * cos_expanded[..., 0] - x2 * sin_expanded[..., 0],
            x1 * sin_expanded[..., 0] + x2 * cos_expanded[..., 0]
        ], dim=-1)
        
        # 원래 형태로 복원
        result = x_rotated.view(batch_size, seq_len, d_model)
        
        # 단일 시퀀스인 경우 원래 형태로 복원
        if single_seq:
            result = result.squeeze(0)
            
        return result

class ResidualMamba_Block(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, drop_p):
        super(ResidualMamba_Block, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(drop_p)
        
    def forward(self, x):
        h = self.norm(x)
        h = self.mamba(h)
        h = self.drop(h)
        return h + x

class Mamba_Encoder(nn.Module):
    def __init__(self, 
                 state_dim, 
                 context_length,
                 d_model, 
                 d_state, 
                 d_conv, 
                 expand,
                 n_layers,
                 drop_p):
        """
        Mamba-based encoder with RoPE for Offline RL with D4RL datasets
        Similar to Decision Transformer: takes state sequence and outputs fixed-dim embedding
        
        Args:
            state_dim: Dimension of state vectors
            context_length: Length of context sequence (L)
            d_model: Model dimension for Mamba
            d_state: SSM state expansion factor
            d_conv: Local convolution width
            expand: Block expansion factor
            n_layers: Number of Mamba blocks
        """
        super(Mamba_Encoder, self).__init__()
        
        self.state_dim = state_dim
        self.context_length = context_length
        self.d_model = d_model
        
        # Input projection: state_dim -> d_model
        self.input_proj = nn.Linear(state_dim, d_model)
        
        # RoPE for positional information
        self.rope = RoPE(d_model, max_seq_len=context_length)
        
        # Mamba block for sequence modeling with layer normalization and dropout
        self.mamba_blocks = nn.ModuleList([
            ResidualMamba_Block(d_model, d_state, d_conv, expand, drop_p)
            for _ in range(n_layers)
        ])
        
        self.post_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

        # Attention pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, states, attention_mask=None):
        if states.dim() == 2:
            states = states.unsqueeze(0)
            single_seq = True
        else:
            single_seq = False

        B, L, _ = states.shape
        if self.context_length is not None and L != self.context_length:
            raise ValueError(f"Expected sequence length {self.context_length}, got {L}")

        x = self.input_proj(states)
        x = self.rope(x, L)

        for layer in self.mamba_blocks:
            x = layer(x)

        x = self.post_norm(x)
        x = self.output_proj(x)

        if single_seq:
            x = x.squeeze(0)
        return x
    
    def encode_trajectory(self, states, mask=None):
        """
        Encode a trajectory of states and return aggregated representation
        Uses attention-weighted pooling to leverage full sequence information
        
        Args:
            states: Input states of shape (batch_size, context_length, state_dim)
                   or (context_length, state_dim) for single sequence
            mask: Optional mask of shape (batch_size, context_length) where 1=valid, 0=pad
        
        Returns:
            final_encoding: Aggregated encoded representation of shape (batch_size, d_model)
                           or (d_model,) for single sequence
        """
        encoded = self.forward(states)  # (B,L,d) 또는 (L,d)
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(0)
            single = True
        else:
            single = False

        if mask is None:
            # 기존 attention-weighted pooling
            att = self.attention_pooling(encoded)            # (B,L,1)
            w = torch.softmax(att, dim=1)                    # (B,L,1)
            z = (encoded * w).sum(dim=1)                     # (B,d)
        else:
            # mask: (B,L) 1=valid, 0=pad
            m = mask.unsqueeze(-1)                           # (B,L,1)
            # (선택) 어텐션 점수에도 -inf 마스킹
            att = self.attention_pooling(encoded).squeeze(-1)  # (B,L)
            att = att.masked_fill(m.squeeze(-1)==0, float('-inf'))
            w = torch.softmax(att, dim=1).unsqueeze(-1)      # (B,L,1)
            z = (encoded * w * m).sum(dim=1) / (m.sum(dim=1).clamp(min=1e-6))  # (B,d)

        if single: 
            z = z.squeeze(0)
        return z
    
    def encode_last_valid(self, states, mask):
        """
        마지막 유효 스텝의 latent representation 추출 (KNN/InfoNCE/consistency에 추천)
        
        Args:
            states: Input states of shape (batch_size, context_length, state_dim)
            mask: Mask of shape (batch_size, context_length) where 1=valid, 0=pad
        
        Returns:
            last_valid: Last valid step encoding of shape (batch_size, d_model)
        """
        # mask: (B,L) → 마지막 1의 인덱스
        x = self.forward(states)             # (B,L,d)
        lengths = mask.sum(dim=1).clamp(min=1).long()  # (B,)
        idx = (lengths - 1).view(-1,1,1).expand(-1,1,x.size(-1))  # (B,1,d)
        last = x.gather(1, idx).squeeze(1)   # (B,d)
        return last
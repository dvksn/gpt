import torch.nn as nn
import torch
from torch.nn import functional as F
import PyPDF2

d_model = 512 # vocab dimenstion
block_size = 256 # context length
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
dff = 2048  ## feed forward network dim
## attention
num_heads = 8
num_attn_layers = 6
dropout = 0.1
## adam optimizerf param
beta1 = 0.9
beta2 = 0.98
epsilon  = 1e-9
warmup_steps = 4000
## maximum training iterations
max_iter = 5000
## 100 sample to estimate the loss
eval_iters = 100
## evaluate the model every 50 iterations
eval_interval = 50
learning_rate = 3e-4
torch.manual_seed(1337)

class Tokenizer(nn.Module):
    def __init__(self, char_list):
        super().__init__()
        self.str_to_int = dict(zip(char_list, range(0, len(char_list))))
        self.int_to_str = dict(zip(range(0, len(char_list)), char_list))
    encode = lambda self, word: [self.str_to_int[ch] for ch in word]
    decode = lambda self, list_num: "".join([self.int_to_str[num] for num in list_num])


def get_batch(split):
    data = train if split == "train" else val
    ## choose a random set of indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) ## save to cuda to speed up the training
    return x, y

@torch.no_grad() ## we are callinf model functions but don't want to update the weights
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            ##print("input shape from estimate_loss: {}".format(X.shape))
            ##print("target shape from estimate_loss: {}".format(Y.shape))
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class SelfAttention(nn.Module):
    def __init__ (self, is_causal):
        super().__init__()
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal

    def forward(self, x):
        B, T, n_emd = x.shape ## n_emb should be same as d_model
        key = self.key(x) ## BxTxd_head
        query = self.query(x) ## BxTxd_head
        value = self.value(x)
        attention = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5 ## BxTxT : t= block_size
        if (self.is_causal):
            attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        hidden_state = attention @ value ## BxTxT BxTxH 
        return hidden_state

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, is_causal):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(is_causal) for _ in range(num_heads)]) ## num_heads x BxTxH
        self.proj = nn.Linear(num_heads*d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.concat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        ### questions> how the order is decided for relu and dropout
        self.ffn = nn.Sequential(nn.Linear(d_model, dff),
                                 nn.ReLU(),
                                 nn.Linear(dff, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.ffn(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(num_heads, True)
        self.ffa = FeedForwardNetwork()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        mhsa_out = self.mhsa(x)  ## BxTxd_model
        ln_out = self.ln1(mhsa_out + x) ## BxTxd_model
        ff_out = self.ffa(ln_out) ## BxTxd_model
        ln_out2 = self.ln2(ff_out+ln_out) ## BxTxd_model
        return ln_out2
    ## this is the update from the original paper where layernorm is applied before self attention
    # def forward(self, x):
    #         x = x + self.sa(self.ln1(x))
    #         x = x + self.ffwd(self.ln2(x))
    #         return x

class Transformer(nn.Module):
    def __init__ (self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len_vocab, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.attn_blocks = nn.Sequential(*[Block() for _ in range(num_attn_layers)])
        self.linear = nn.Linear(d_model, len_vocab)

    def forward(self, x, y=None):
        B, T = x.shape
        ##print("shape of input {0}".format(T))
        token_embedding = self.token_embedding_table(x) ## BxTxd_model
        ##print("shape of input {0}".format(token_embedding.shape))
        ##print("shape of pos {0}".format(device))
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        ##print("shape of input {0}".format(pos_emb.shape))
        input = token_embedding + pos_emb
        block_op = self.attn_blocks(input)
        logits = self.linear(block_op) ## BxTxvocab_len
        if (y == None):
            loss = None
        else:
            ## y: B,T 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, input, max_new_tokens):
        for i in range(max_new_tokens):
            ## for forward network we need BxT shape only last T tokens
            context = input[:,-block_size:]
            logits, loss = self(context) ## logits B
            logits = logits[:,-1,:]
            prob = F.softmax(logits, dim=-1)
            next_ch = torch.multinomial(prob, num_samples=1) # Bx1
            input = torch.concat([input, next_ch], dim=1)
        return context


if __name__ == "__main__":
    # with open('input.txt', 'r', encoding='utf-8') as f:
    #     text = f.read()
    all_books = list()
    for f in ["Steve Jobs.pdf", "9781405293181.pdf", "Eric-Jorgenson_The-Almanack-of-Naval-Ravikant_Final.pdf"]:
        reader = PyPDF2.PdfReader("C:\\Users\\sonalimittal\\OneDrive - Microsoft\\Desktop\\Books\\{0}".format(f))
        num_pages =  len(reader.pages)
        book_content = [reader.pages[i].extract_text() for i in range(0,num_pages)]
        all_books.append(" ".join(book_content))
    text = " ".join(all_books)
    
    ## tokenization at charater level
    char_list = sorted(list(set(text)))
    len_vocab = len(char_list)
    tkn = Tokenizer(char_list)
    text = torch.tensor(tkn.encode(text), dtype=torch.long)
    train_size = int(0.9*len(text))
    train = text[:train_size]
    val = text[train_size:]
    print(len_vocab)
    print(char_list)
    if (d_model%num_heads!=0):
        print("Provide correct value for d_model")
    
    d_head = d_model//num_heads
    
    tmodel = Transformer()
    model = tmodel.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    ## we provide params so that optimizer can update the gradient using optimizer.step function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
    for iter in range(max_iter):
        if (iter%eval_interval == 0):
            losses = estimate_loss()
            print(f"iter {iter}, train loss: {losses['train']}, val loss: {losses['val']}")
        x, y = get_batch("train")

        # evaluate the loss
        logits, loss = model(x, y) ## here weights will be updated
        ## In pytorch every parameter has a grad attributes which is used to store gradients 
        ## pytorch accumulates the gradient so we need to set the gradients to zero before any backward pass
        ## The accumulation should happen within a batch.
        optimizer.zero_grad(set_to_none=True) 
        ## loss.backward will calculate the gradient for every param with required_grad = True
        loss.backward()
        ## optimizer will update the parameters : W = W - learning_rate*grad
        optimizer.step()
        if (iter%50 == 0):
            torch.save({
                'epoch': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, 'model_{0}.pth'.format(iter))
    torch.save(model.state_dict(), 'model_weights.pth')
    torch.save(model, 'model.pth')
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    ans = tkn.decode(model.generate(context, max_new_tokens=500)[0].tolist())
    with open("output.txt", "w") as f:
        f.write(ans)
    print(ans)
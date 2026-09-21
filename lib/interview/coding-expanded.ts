import {b,type Question,type Case} from './schema';
function task(id:number,topic:string,title:string,enTitle:string,prompt:string,enPrompt:string,signature:string,solution:string,tests:Case[],why:string,enWhy:string,complexity:string):Question{return {id:String(id),topic,kind:'code',difficulty:[993,997].includes(id)?'hard':[984,990,995,996,998,999,1000].includes(id)?'easy':'medium',title:b(title,enTitle),prompt:b(prompt,enPrompt),starter:`def ${signature}:\n    pass\n`,solution,tests,explanation:b(why,enWhy),complexity:b(complexity,complexity),followup:b('输入规模增大或改为低精度时，哪些假设需要重新验证？','Which assumptions need rechecking at larger scale or lower precision?'),sourceIds:[topic==='evaluation'?'helm':topic==='rag'?'rag':topic==='multimodal'?'mae':'basics']}}
const c=(expression:string,expected:unknown):Case=>({expression,expected});
export const extraCoding:Question[]=[
task(981,'foundations','稳定 LogSumExp','Stable log-sum-exp','实现 logsumexp(xs)，输入为非空有限实数列表；返回 log(Σexp(x))，大数不得上溢。空列表抛 ValueError。','Implement logsumexp(xs) for a nonempty finite-real list, returning log(sum(exp(x))) without overflow. Raise ValueError for empty input.','logsumexp(xs)',`import math
def logsumexp(xs):
    if not xs: raise ValueError('empty')
    m=max(xs)
    return m+math.log(sum(math.exp(x-m) for x in xs))
`,[c('logsumexp([0])',0),c('logsumexp([0,0])',Math.log(2)),c('logsumexp([1000,1000])',1000+Math.log(2)),c('logsumexp([-1000])',-1000),{expression:'logsumexp([])',raises:'ValueError'}],'先减最大值控制指数范围，再把它加回 log 域。这里的平移影响最终值，不能像 softmax 一样省略加回步骤。','Subtract the maximum before exponentiation, then add it back in log space. Unlike softmax, the shift affects the returned value.','Time O(n), space O(1).'),
task(982,'architecture','一维 LayerNorm','One-dimensional LayerNorm','实现 layernorm(xs, eps=1e-5)，使用总体方差（除以 n），不含可学习仿射参数。输入非空有限实数，eps>0；空输入抛 ValueError。','Implement layernorm(xs, eps=1e-5), using population variance divided by n, without affine parameters. Finite inputs, eps>0; raise ValueError for empty input.','layernorm(xs, eps=1e-5)',`import math
def layernorm(xs, eps=1e-5):
    if not xs: raise ValueError('empty')
    mean=sum(xs)/len(xs)
    var=sum((x-mean)**2 for x in xs)/len(xs)
    return [(x-mean)/math.sqrt(var+eps) for x in xs]
`,[c('layernorm([2,2])',[0,0]),c('layernorm([1,3], 3)',[-.5,.5]),c('layernorm([7])',[0]),c('layernorm([-3,-1],3)',[-.5,.5]),{expression:'layernorm([])',raises:'ValueError'}],'先算均值，再算偏离均值的平方均值。epsilon 放在平方根里面；总体方差与统计学无偏样本方差不同。','Compute the mean and then mean squared deviations. Epsilon is inside the square root; population variance differs from unbiased sample variance.','Time O(n), output O(n).'),
task(983,'architecture','一维 RMSNorm','One-dimensional RMSNorm','实现 rmsnorm(xs,eps=1e-5)，不减均值、不含仿射参数。输入为非空有限列表，eps>0；空输入抛 ValueError。','Implement rmsnorm(xs,eps=1e-5) without centering or affine parameters. Nonempty finite input and eps>0; raise ValueError for empty input.','rmsnorm(xs, eps=1e-5)',`import math
def rmsnorm(xs, eps=1e-5):
    if not xs: raise ValueError('empty')
    scale=math.sqrt(sum(x*x for x in xs)/len(xs)+eps)
    return [x/scale for x in xs]
`,[c('rmsnorm([0,0])',[0,0]),c('rmsnorm([1,1],3)',[.5,.5]),c('rmsnorm([-1,1],3)',[-.5,.5]),c('rmsnorm([2],12)',[.5]),{expression:'rmsnorm([])',raises:'ValueError'}],'RMSNorm 缩放的是原始向量。相同常量向量经 RMSNorm 后通常非零，这正是它与 LayerNorm 的区别。','RMSNorm rescales the original vector. A constant nonzero vector usually remains nonzero, unlike LayerNorm.','Time O(n), output O(n).'),
task(984,'foundations','离散熵','Discrete entropy','实现 entropy(ps)，自然对数，0·log(0) 按 0。输入为有效概率列表（和为 1）；约定空列表返回 0。','Implement entropy(ps) with natural logs and 0*log(0)=0. Input is a valid probability list summing to one; define empty input as zero.','entropy(ps)',`import math
def entropy(ps):
    return -sum(p*math.log(p) for p in ps if p>0)
`,[c('entropy([1])',0),c('entropy([0.5,0.5])',Math.log(2)),c('entropy([0,1,0])',0),c('entropy([0.25]*4)',Math.log(4)),c('entropy([])',0)],'零概率事件的贡献按极限定义为零，因此必须跳过 log(0)。熵只描述这份分布，不能证明模型预测正确。','Zero-probability terms contribute zero by the limit definition, so skip log(0). Entropy describes this distribution, not prediction correctness.','Time O(n), space O(1).'),
task(985,'foundations','有限支持 KL','KL on finite support','实现 kl(p,q)=Σp·log(p/q)，输入等长有效分布。忽略 p=0 的项；p>0 且 q=0 抛 ValueError。空列表对返回 0。','Implement kl(p,q)=sum(p*log(p/q)) for equally sized valid distributions. Skip p=0; raise ValueError if p>0 and q=0. Empty lists return zero.','kl(p, q)',`import math
def kl(p,q):
    total=0.0
    for a,b in zip(p,q):
        if a>0:
            if b==0: raise ValueError('unsupported event')
            total+=a*math.log(a/b)
    return total
`,[c('kl([0.5,0.5],[0.5,0.5])',0),c('kl([1,0],[0.5,0.5])',Math.log(2)),c('kl([0,1],[0,1])',0),c('kl([],[])',0),{expression:'kl([1,0],[0,1])',raises:'ValueError'}],'KL 的方向固定为 p 到 q。这里用异常表示支持集不满足而非无穷值，便于浏览器判题；不要悄悄加平滑改变问题。','The direction is p to q. This contract raises on support violations instead of returning infinity for browser grading; do not silently smooth the distributions.','Time O(n), space O(1).'),
task(986,'training','从 Logits 算交叉熵','Cross-entropy from logits','实现 cross_entropy(logits,target)，非空有限 logits，target 是合法索引；返回单样本负对数概率，不先把目标概率四舍五入。','Implement cross_entropy(logits,target) for nonempty finite logits and a valid target index. Return negative log probability without rounding target probabilities.','cross_entropy(logits, target)',`import math
def cross_entropy(logits,target):
    m=max(logits)
    return m+math.log(sum(math.exp(x-m) for x in logits))-logits[target]
`,[c('cross_entropy([0,0],0)',Math.log(2)),c('cross_entropy([1000,1000],1)',Math.log(2)),c('cross_entropy([5],0)',0),c('cross_entropy([0,1],0)',Math.log(1+Math.E)),c('cross_entropy([0,1000],0)',1000)],'负对数概率等于 logsumexp(logits)−目标 logit。直接在 log 域计算，可避免目标 softmax 概率下溢为零后再取 log。','Negative log probability is logsumexp minus the target logit. Working in log space avoids taking log of an underflowed probability.','Time O(n), space O(1).'),
task(987,'training','全局范数裁剪','Global-norm clipping','实现 clip_norm(xs,limit)，limit>0，有限输入。若 L2 范数超过 limit，所有元素乘同一因子，否则返回相同数值的新列表；不修改输入。','Implement clip_norm(xs,limit), with limit>0 and finite inputs. Scale all elements together if L2 norm exceeds the limit; otherwise return a new equal list. Do not mutate input.','clip_norm(xs, limit)',`import math
def clip_norm(xs,limit):
    norm=math.sqrt(sum(x*x for x in xs))
    scale=min(1,limit/norm) if norm else 1
    return [x*scale for x in xs]
`,[c('clip_norm([3,4],1)',[.6,.8]),c('clip_norm([3,4],10)',[3,4]),c('clip_norm([0,0],1)',[0,0]),c('clip_norm([],1)',[]),c('clip_norm([-3,4],2)',[-1.2,1.6])],'缩放因子 min(1,limit/norm) 保持方向。零范数需要避免除零；本题不是逐元素 clamp。','The common factor min(1,limit/norm) preserves direction. Avoid division by zero for zero norm; this is not elementwise clamping.','Time O(n), output O(n).'),
task(988,'serving','Top-k 概率过滤','Top-k probability filtering','实现 topk_probs(ps,k)，保留概率最高的 min(k,n) 项并重新归一化，其余为 0。并列按原索引小者优先；k 为正整数，非空 ps 非负且总和>0；空输入返回 []。','Implement topk_probs(ps,k): retain min(k,n) largest probabilities and renormalize, zeroing others. Break ties by lower index. Positive integer k, nonnegative nonempty ps with positive total; empty returns [].','topk_probs(ps, k)',`def topk_probs(ps,k):
    ids=sorted(range(len(ps)),key=lambda i:(-ps[i],i))[:k]
    total=sum(ps[i] for i in ids)
    keep=set(ids)
    return [p/total if i in keep else 0 for i,p in enumerate(ps)]
`,[c('topk_probs([0.5,0.3,0.2],2)',[.625,.375,0]),c('topk_probs([1,1,1],1)',[1,0,0]),c('topk_probs([1,2],9)',[1/3,2/3]),c('topk_probs([],2)',[]),c('topk_probs([0,1,0],2)',[0,1,0])],'排序时显式写并列规则，过滤后重新归一化。此题返回过滤分布，不随机抽样，因此结果可以确定性验证。','Specify tie-breaking and renormalize after filtering. Return the filtered distribution rather than a random sample for deterministic validation.','Time O(n log n), space O(n).'),
task(989,'serving','Nucleus 过滤','Nucleus filtering','实现 nucleus(ps,p)，ps 是非空有效概率分布，0<p≤1。降序取累计概率首次达到 p 的最短前缀，并归一化后还原原顺序；并列按索引。空输入返回 []。','Implement nucleus(ps,p) for a valid probability distribution and 0<p<=1. Keep the shortest descending prefix reaching p, renormalize and restore original order; ties use index. Empty returns [].','nucleus(ps, p)',`def nucleus(ps,p):
    ids=sorted(range(len(ps)),key=lambda i:(-ps[i],i))
    keep=[]
    total=0.0
    for i in ids:
        keep.append(i)
        total+=ps[i]
        if total>=p: break
    chosen=set(keep)
    return [x/total if i in chosen else 0 for i,x in enumerate(ps)]
`,[c('nucleus([0.5,0.3,0.2],0.7)',[.625,.375,0]),c('nucleus([0.5,0.3,0.2],0.1)',[1,0,0]),c('nucleus([0.5,0.5],0.5)',[1,0]),c('nucleus([0.2,0.8],1)',[.2,.8]),c('nucleus([],0.9)',[])],'必须包含使累计概率跨过阈值的那一项。排序只用于确定保留集合，输出位置仍对应原始 token。','Include the item crossing the threshold. Sorting selects retained indices; output positions must still correspond to original tokens.','Time O(n log n), space O(n).'),
task(990,'foundations','BPE 单次合并','One BPE merge','实现 merge_pair(tokens,a,b)，从左到右将非重叠相邻 a,b 合并为字符串 a+b。不反复对新结果再合并，不修改输入。','Implement merge_pair(tokens,a,b), replacing nonoverlapping adjacent a,b pairs left-to-right with string a+b. Do not recursively merge new results or mutate input.','merge_pair(tokens, a, b)',`def merge_pair(tokens,a,b):
    out=[]
    i=0
    while i<len(tokens):
        if i+1<len(tokens) and tokens[i]==a and tokens[i+1]==b:
            out.append(a+b)
            i+=2
        else:
            out.append(tokens[i])
            i+=1
    return out
`,[c("merge_pair(['a','b','a','b'],'a','b')",['ab','ab']),c("merge_pair(['a','a','a'],'a','a')",['aa','a']),c("merge_pair([],'a','b')",[]),c("merge_pair(['b','a'],'a','b')",['b','a']),c("merge_pair(['a','b','b'],'a','b')",['ab','b'])],'扫描指针匹配后前进两位，避免同一 token 被使用两次。本题是完整 BPE 编码中的一个局部操作，不是完整 tokenizer。','Advance by two after a match to avoid consuming a token twice. This is one operation inside BPE, not a complete tokenizer.','Time O(n), output O(n).'),
task(991,'architecture','二维 RoPE 旋转','Two-dimensional RoPE rotation','实现 rotate2(x,angle)，x 为两个有限实数组成的列表，angle 为弧度。返回 [x0 cos−x1 sin, x0 sin+x1 cos]。','Implement rotate2(x,angle) for a two-element finite vector and angle in radians, returning [x0*cos-x1*sin, x0*sin+x1*cos].','rotate2(x, angle)',`import math
def rotate2(x,angle):
    c,s=math.cos(angle),math.sin(angle)
    return [x[0]*c-x[1]*s,x[0]*s+x[1]*c]
`,[c('rotate2([1,0],0)',[1,0]),c('rotate2([1,0],1.5707963267948966)',[0,1]),c('rotate2([0,1],1.5707963267948966)',[-1,0]),c('rotate2([0,0],2)',[0,0]),c('rotate2([3,4],3.141592653589793)',[-3,-4])],'二维旋转保持范数，但会改变方向。注意弧度以及 sin 项的正负；多维 RoPE 将不同频率应用到多个二维子空间。','Rotation preserves norm while changing direction. Check radians and sine signs; multidimensional RoPE applies different frequencies across two-dimensional subspaces.','Time O(1), space O(1).'),
task(992,'architecture','因果 Padding 联合掩码','Combined causal/padding mask','实现 attention_mask(lengths,width)，返回 B×width×width 的 bool 列表。只有 query 和 key 都小于该样本长度且 key≤query 时为 True。lengths 为 0..width 的整数，width≥0。','Implement attention_mask(lengths,width), returning B×width×width bool lists. True iff query and key are valid for the sample and key<=query. Lengths are integers in 0..width; width>=0.','attention_mask(lengths, width)',`def attention_mask(lengths,width):
    return [[[i<n and j<n and j<=i for j in range(width)] for i in range(width)] for n in lengths]
`,[c('attention_mask([2],2)',[[[true,false],[true,true]]]),c('attention_mask([1],2)',[[[true,false],[false,false]]]),c('attention_mask([0],1)',[[[false]]]),c('attention_mask([],2)',[]),c('attention_mask([0],0)',[[]])],'本题也屏蔽无效 query 行。注意某些 attention kernel 对全 mask 行有特殊数值约定，不能直接把此布尔结果当作 softmax 的最终安全实现。','This contract masks invalid query rows too. Some attention kernels need special handling for fully masked rows; a Boolean mask alone is not a complete safe softmax implementation.','Time and output O(B·width²).'),
task(993,'hardware','合并 Softmax 统计量','Merge softmax statistics','实现 merge_stats(a,b)，每个非空块的统计为 [m,s]，m 是最大值，s=Σexp(x−m)。返回合并后的 [m,s]。两块都非空且统计有限。','Implement merge_stats(a,b), where each nonempty block has [m,s], m=max(x), s=sum(exp(x-m)). Return merged [m,s]. Both blocks are nonempty with finite statistics.','merge_stats(a, b)',`import math
def merge_stats(a,b):
    m=max(a[0],b[0])
    return [m,a[1]*math.exp(a[0]-m)+b[1]*math.exp(b[0]-m)]
`,[c('merge_stats([0,1],[0,1])',[0,2]),c('merge_stats([1,1],[0,1])',[1,1+Math.exp(-1)]),c('merge_stats([1000,2],[1000,3])',[1000,5]),c('merge_stats([-1,1],[-2,2])',[-1,1+2*Math.exp(-1)]),c('merge_stats([0,2],[1000,1])',[1000,1])],'两个 s 使用不同的参考最大值，不能直接相加。先换到共同的较大最大值，再合并缩放后的和，这是在线 softmax 的核心。','The sums use different reference maxima and cannot simply be added. Rescale both to the larger maximum before summing: the core of online softmax.','Time O(1), space O(1).'),
task(994,'rag','Masked Mean Pooling','Masked mean pooling','实现 mean_pool(rows,mask)。rows 是非空等宽有限矩阵，mask 等长 bool 列表。仅平均 True 行；无有效行时返回该宽度的零列表。','Implement mean_pool(rows,mask) for a nonempty rectangular finite matrix and equally long Boolean mask. Average only True rows; return a zero vector if none are valid.','mean_pool(rows, mask)',`def mean_pool(rows,mask):
    count=sum(mask)
    if not count: return [0.0]*len(rows[0])
    return [sum(row[j] for row,valid in zip(rows,mask) if valid)/count for j in range(len(rows[0]))]
`,[c('mean_pool([[1,2],[3,4]],[True,True])',[2,3]),c('mean_pool([[1,2],[99,99]],[True,False])',[1,2]),c('mean_pool([[1,2]],[False])',[0,0]),c('mean_pool([[5]],[True])',[5]),c('mean_pool([[],[]],[True,True])',[])],'分母是有效行数，不是总行数。全 mask 的零向量是本题接口约定；后续 cosine 必须处理零范数。','Divide by valid-row count, not total rows. The all-masked zero vector is a contract choice; downstream cosine must handle zero norm.','Time O(n·d), output O(d).'),
task(995,'evaluation','Mean Reciprocal Rank','Mean reciprocal rank','实现 mrr(rankings)，输入为多个 bool 列表，True 表示该位置相关。每个列表贡献首个 True 的 1-based 倒数排名，无命中为 0；外层空返回 0。','Implement mrr(rankings) over Boolean relevance lists. Each contributes reciprocal one-based rank of the first True, or zero for no hit; empty outer input returns zero.','mrr(rankings)',`def mrr(rankings):
    if not rankings: return 0.0
    total=0.0
    for row in rankings:
        for i,hit in enumerate(row,1):
            if hit:
                total+=1/i
                break
    return total/len(rankings)
`,[c('mrr([[True],[False,True]])',.75),c('mrr([[False,False]])',0),c('mrr([])',0),c('mrr([[],[True]])',.5),c('mrr([[False,False,True,True]])',1/3)],'MRR 只看首个相关结果，不累加之后的命中。没有相关项的查询仍计入平均分母。','MRR uses only the first relevant result; later hits do not add credit. No-hit queries remain in the averaging denominator.','Time O(total ranked items), space O(1).'),
task(996,'rag','去重 Recall@k','Deduplicated recall@k','实现 recall_at_k(ranked,relevant,k)，ID 为字符串。只看 ranked 的前 max(k,0) 个位置，对命中 ID 去重后除以 relevant 的唯一 ID 数；无相关 ID 返回 0。','Implement recall_at_k(ranked,relevant,k) for string IDs. Inspect the first max(k,0) positions, deduplicate hits and divide by unique relevant IDs; no relevant IDs returns zero.','recall_at_k(ranked, relevant, k)',`def recall_at_k(ranked,relevant,k):
    truth=set(relevant)
    if not truth: return 0.0
    return len(set(ranked[:max(k,0)]) & truth)/len(truth)
`,[c("recall_at_k(['a','b'],['a','b','c'],2)",2/3),c("recall_at_k(['a','a'],['a','b'],2)",.5),c("recall_at_k(['a'],[],1)",0),c("recall_at_k(['a'],['a'],-1)",0),c("recall_at_k(['b','a'],['a','a'],1)",0)],'先按排名截断，再对命中去重，避免重复文档刷高分。相关集合去重后作为分母。','Truncate by rank before deduplicating hits so duplicates cannot inflate recall. Deduplicate the relevant set for the denominator.','Time O(k+r), space O(k+r).'),
task(997,'evaluation','Pass@k 无偏估计式','Pass@k estimator','实现 pass_at_k(n,c,k)=1−C(n−c,k)/C(n,k)。n≥1，0≤c≤n，1≤k≤n，均为整数。若 n−c<k，返回 1。','Implement pass_at_k(n,c,k)=1-C(n-c,k)/C(n,k), for integers n>=1, 0<=c<=n, 1<=k<=n. Return one if n-c<k.','pass_at_k(n, c, k)',`def pass_at_k(n,c,k):
    if n-c<k: return 1.0
    miss=1.0
    for i in range(k): miss*=(n-c-i)/(n-i)
    return 1-miss
`,[c('pass_at_k(10,1,1)',.1),c('pass_at_k(10,1,2)',.2),c('pass_at_k(5,0,3)',0),c('pass_at_k(5,5,1)',1),c('pass_at_k(5,2,2)',.7)],'用连乘计算全部 k 个样本都失败的概率，避免巨大组合数。该估计式依赖明确采样协议，不能把它直接当作任意相关重试的真实成功概率。','Compute the all-fail probability as a product rather than giant binomial coefficients. The estimator depends on a defined sampling protocol, not arbitrary correlated retries.','Time O(k), space O(1).'),
task(998,'evaluation','Nearest-rank 分位数','Nearest-rank percentile','实现 percentile(xs,p)，非空有限列表，0<p≤1。排序后返回 ceil(p·n) 对应的 1-based 元素，不插值且不修改输入；空输入抛 ValueError。','Implement percentile(xs,p), for nonempty finite xs and 0<p<=1. Return the one-based ceil(p*n) order statistic without interpolation or mutation; empty input raises ValueError.','percentile(xs, p)',`import math
def percentile(xs,p):
    if not xs: raise ValueError('empty')
    return sorted(xs)[math.ceil(p*len(xs))-1]
`,[c('percentile([4,1,3,2],0.5)',2),c('percentile([4,1,3,2],1)',4),c('percentile([7],0.95)',7),c('percentile([-2,-1,0],0.1)',-2),{expression:'percentile([],0.5)',raises:'ValueError'}],'不同库可能采用不同插值定义，本题明确 nearest-rank。比较服务 p99 前必须先统一分位数算法和失败样本处理。','Libraries use different interpolation conventions; this task explicitly uses nearest rank. Align percentile algorithms and failure accounting before comparing service p99.','Time O(n log n), space O(n).'),
task(999,'multimodal','只算隐藏像素的 MSE','Masked-only pixel MSE','实现 masked_mse(pred,target,mask)，等长有限列表，mask=True 的位置才计入。无选中位置返回 None，而不是 0。','Implement masked_mse(pred,target,mask) for equally long finite lists. Include only mask=True positions; return None when no positions are selected, not zero.','masked_mse(pred, target, mask)',`def masked_mse(pred,target,mask):
    errors=[(p-t)**2 for p,t,m in zip(pred,target,mask) if m]
    return sum(errors)/len(errors) if errors else None
`,[c('masked_mse([1,3],[0,1],[True,True])',2.5),c('masked_mse([99,3],[0,1],[False,True])',4),c('masked_mse([1],[1],[True])',0),c('masked_mse([1],[0],[False])',null),c('masked_mse([],[],[])',null)],'只在 mask 内累计误差并用隐藏像素数归一化。没有评测像素意味着指标未定义，不能显示成完美重建。','Accumulate and normalize only over masked pixels. No evaluated pixels means undefined error, not perfect reconstruction.','Time O(n), space O(n) in this reference.'),
task(1000,'rag','带零范数契约的 Cosine','Cosine with zero-norm semantics','实现 cosine(a,b)，等长有限列表。非零向量按标准余弦公式；任一零范数（包括空输入）返回 None。不修改输入。','Implement cosine(a,b) for equally long finite lists. Use standard cosine for nonzero vectors; return None for either zero norm, including empty input. Do not mutate inputs.','cosine(a, b)',`import math
def cosine(a,b):
    na=math.sqrt(sum(x*x for x in a))
    nb=math.sqrt(sum(x*x for x in b))
    if not na or not nb: return None
    return sum(x*y for x,y in zip(a,b))/(na*nb)
`,[c('cosine([1,0],[1,0])',1),c('cosine([1,0],[-1,0])',-1),c('cosine([1,0],[0,1])',0),c('cosine([0,0],[1,2])',null),c('cosine([],[])',null)],'先检查两个范数，再除以它们的乘积。零向量返回 None 是显式接口选择，避免悄悄把未定义相似度误当正交。','Check both norms before dividing. Returning None for zero vectors is an explicit contract, avoiding mistaking undefined similarity for orthogonality.','Time O(n), space O(1).')
];

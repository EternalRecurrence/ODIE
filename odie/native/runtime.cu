// Shared native TCNN training runtime. Python owns sampling and stage activation.
// Model/storage constants are passed by CMake and included in the build cache key.
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <json/json.hpp>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "kernels.cuh"
#include <memory>
#include <string>
using json = nlohmann::json;
using H = tcnn::network_precision_t;
using Mat = tcnn::GPUMatrixDynamic<H>;
template<class T> using GPU = tcnn::GPUMemory<T>;
constexpr int L=ODIE_LEVELS, RAW=ODIE_RAW_WIDTH, W=((RAW+15)/16)*16, G=((L+15)/16)*16, B=65536, TILE=256;
static_assert(L>=2 && L<=20, "Native router supports 2..20 levels");
inline int blocks(int n) { return (n+255)/256; }
template<class T> T* ptr(const json& j,const char* key) { return reinterpret_cast<T*>(j.at(key).get<uint64_t>()); }

__device__ int row_of(const int* d, int node,const uint64_t* bitmap,const int* prefix) {
    if(d[7]) return node;
    int word=d[3]+(node>>6), bit=node&63;
    return prefix[word]+__popcll(bitmap[word]&((uint64_t{1}<<bit)-1));
}
__device__ bool cell_on(const int* d,int cell,const uint8_t* bitmap) {
    return (bitmap[d[4]+(cell>>3)]>>(cell&7))&1;
}
__global__ void input_batch(const int64_t* ids,const uint8_t* rgb,float* xy,float* target,int n,int real,int width,int height,bool fused) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    uint32_t id=i<real?ids[i]:0;
    // PyTorch scalar division multiplies by the FP32 reciprocal.
    xy[2*i]=__fmul_rn(float(id%width),float(1.0/double(width-1)));
    xy[2*i+1]=__fmul_rn(float(id/width),float(1.0/double(height-1)));
    if(!fused)for(int c=0;c<3;++c)target[3*i+c]=(ODIE_RGB8 ? __fmul_rn(float(rgb[3ull*id+c]),1.f/255.f) : reinterpret_cast<const float*>(rgb)[3ull*id+c]);
}
__global__ void query(const float* xy,const float* grid,const int* desc,const uint64_t* bits,const int* prefix,
                      const uint8_t* cells,H* raw,uint8_t* visibility,int n,int active) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,l=blockIdx.y;
    if(i>=n)return;
    const int* d=desc+8*l; int cell;float w[4];
    odie::geometry(xy[2*i],xy[2*i+1],d[0],cell,w);
    bool on=l<active && cell_on(d,cell,cells);visibility[i*L+l]=on;
    for(int c=0;c<d[5];++c){H v=__float2half_rn(0);
        if(on)for(int k=0;k<4;++k){int node=(cell+(k&1)+(k>>1)*d[0])%d[1];
            v=__hfma(__float2half_rn(w[k]),__float2half_rn(grid[d[2]+row_of(d,node,bits,prefix)*d[5]+c]),v);}
        raw[i*W+d[6]+c]=v;
    }
    if(l==0)for(int c=RAW;c<W;++c)raw[i*W+c]=__float2half_rn(1);
}
__global__ void loss_grad(const H* output,const uint8_t* rgb,const float* target,const int64_t* ids,H* dy,float* squares,int n,int real,bool fused) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(i>=real){for(int c=0;c<16;++c)dy[i*16+c]=__float2half_rn(0);squares[i]=0;return;}
    float s=0;for(int c=0;c<3;++c){
        float t=fused?__fmul_rn(float(rgb[3ull*ids[i]+c]),1.f/255.f):target[3*i+c];
        float d=__half2float(output[i*16+c])-t;s+=d*d;
        dy[i*16+c]=__hmul(__float2half_rn((2*d/float(3*real))*128.f),__float2half_rn(128));
    }
    for(int c=3;c<16;++c)dy[i*16+c]=__float2half_rn(0);
    squares[i]=s/3;
}
// Gate features in parallel over samples and levels, with explicit FP16
// rounding of visibility-weighted gates and feature products.
__global__ void parallel_gate(const H* raw,const H* logits,const uint8_t* vis,
    const int* desc,H* weighted,float* sigmoid,int n){
    int item=blockIdx.x*blockDim.x+threadIdx.x;if(item>=n*L)return;
    int i=item/L,l=item%L;const int* d=desc+8*l;
    float z=__half2float(logits[i*G+l]);
    float s=l<ODIE_GATE_K ? 1.f : (ODIE_SOFTPLUS ? (z>20.f ? z : log1pf(expf(z))) : 1.f/(1.f+expf(-z)));sigmoid[item]=s;
    H weight=__float2half_rn(__fmul_rn(s,float(vis[item])));
    for(int c=0;c<d[5];++c){int j=i*W+d[6]+c;weighted[j]=__hmul_rn(raw[j],weight);}
    if(l==L-1)for(int c=RAW;c<W;++c)weighted[i*W+c]=__float2half_rn(1);
}
__global__ void parallel_gate_back(const H* raw,const float* sigmoid,const uint8_t* vis,
    const int* desc,const H* dz,H* direct,H* da,int n,bool fuse_direct){
    int item=blockIdx.x*blockDim.x+threadIdx.x;if(item>=n*L)return;
    int i=item/L,l=item%L;const int* d=desc+8*l;float s=sigmoid[item];
    H weight=__float2half_rn(__fmul_rn(s,float(vis[item]))),reduced=__float2half_rn(0);
    for(int c=0;c<d[5];++c){int j=i*W+d[6]+c;
        H value=__float2half_rn(__half2float(dz[j])/128.f);
        if(!fuse_direct)direct[j]=__hmul_rn(value,weight);
        reduced=__hadd_rn(reduced,__hmul_rn(value,raw[j]));}
    float incoming=__fmul_rn(__half2float(reduced),float(vis[item]));
    float derivative=l<ODIE_GATE_K ? 0.f : (ODIE_SOFTPLUS ? incoming*(-expm1f(-s)) : __fmul_rn(__fmul_rn(incoming,1.f-s),s));
    da[i*G+l]=__hmul_rn(__float2half_rn(derivative),__float2half_rn(128));
    if(l==L-1){
        for(int c=L;c<G;++c)da[i*G+c]=__float2half_rn(0);
        if(!fuse_direct)for(int c=RAW;c<W;++c)direct[i*W+c]=__float2half_rn(0);
    }
}
__global__ void clear_dirty(float* grad,H* hgrad,int* flags,const int* list,const int* count,int total) {
    if(blockIdx.x>=*count)return;
    int tile=list[blockIdx.x],id=tile*TILE+threadIdx.x;
    if(id<total){grad[id]=0;hgrad[id]=__float2half_rn(0);}
    if(threadIdx.x==0)flags[tile]=0;
}
// Direct feature gating and the router's input gradient both feed the encoding.
__global__ void hash_router_gradient(const H* direct,const H* indirect,H* combined,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    combined[i]=__hadd_rn(direct[i],__float2half_rn(__half2float(indirect[i])/128.f));
}
__device__ void mark_dirty(int id,int* flags,int* list,int* count) {
    if(!flags)return;
    int tile=id/TILE;
    auto a=cooperative_groups::coalesced_threads();
    auto g=cooperative_groups::labeled_partition(a,tile);
    if(g.thread_rank()==0 && atomicCAS(flags+tile,0,1)==0)list[atomicAdd(count,1)]=tile;
}
__global__ void grid_back(const float* xy,const H* direct,const H* indirect,const int* desc,
    const uint64_t* bits,const int* prefix,const uint8_t* cells,float* grad,H* hgrad,
    int n,int warp_limit,bool half,int* flags,int* list,int* count,const float* sigmoid,bool fuse_direct) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const int* d=desc+8*blockIdx.y;int cell;float w[4];odie::geometry(xy[2*i],xy[2*i+1],d[0],cell,w);
    if(!cell_on(d,cell,cells))return;
    float values[4];for(int c=0;c<d[5];++c){int j=i*W+d[6]+c;
        H v=direct[j];
#if ODIE_ROUTER_ENABLED
        if(fuse_direct)v=__hmul_rn(__float2half_rn(__half2float(v)/128.f),__float2half_rn(sigmoid[i*L+blockIdx.y]));
        values[c]=__half2float(__hadd_rn(v,__float2half_rn(__half2float(indirect[j])/128.f)));
#else
        values[c]=__half2float(__float2half_rn(__half2float(v)/128.f));
#endif
    }
    if(half && d[5]==2 && d[0]>128){
        if(values[0]==0 && values[1]==0)return;
        for(int k=0;k<4;++k){int node=(cell+(k&1)+(k>>1)*d[0])%d[1];int id=d[2]+2*row_of(d,node,bits,prefix);
            mark_dirty(id,flags,list,count);
            atomicAdd(reinterpret_cast<__half2*>(hgrad+id),__floats2half2_rn(w[k]*values[0],w[k]*values[1]));}
        return;
    }
    for(int c=0;c<d[5];++c){float v=values[c];if(v==0)continue;
        for(int k=0;k<4;++k){int node=(cell+(k&1)+(k>>1)*d[0])%d[1];int id=d[2]+row_of(d,node,bits,prefix)*d[5]+c;
            mark_dirty(id,flags,list,count);
            float add=w[k]*v;
            if(d[0]<=warp_limit){auto a=cooperative_groups::coalesced_threads();auto g=cooperative_groups::labeled_partition(a,id);
                add=cooperative_groups::reduce(g,add,cooperative_groups::plus<float>());
                if(g.thread_rank()==0)atomicAdd(grad+id,add);
            }else atomicAdd(grad+id,add);
        }
    }
}
// Only the coarse F2 and finest F1 ranges need casting. No full FP32 gradient conversion.
__global__ void cast_ranges(const float* grad,H* hgrad,const int* desc,int total) {
    const int* d=desc+8*blockIdx.y;
    if(d[5]==2 && d[0]>128)return;
    int begin=d[2],end=blockIdx.y==L-1?total:desc[(blockIdx.y+1)*8+2];
    for(int i=begin+blockIdx.x*blockDim.x+threadIdx.x;i<end;i+=gridDim.x*blockDim.x)hgrad[i]=__float2half_rn(grad[i]);
}
struct Head {
    std::unique_ptr<tcnn::Network<H>> net;
    std::unique_ptr<tcnn::Optimizer<H>> opt;
    GPU<H> weights,grad;float* master;
    Head(int out,int width,int depth,float* p,float lr):master(p){
        net.reset(tcnn::create_network<H>({{"otype","FullyFusedMLP"},{"n_input_dims",W},{"n_output_dims",out},
            {"n_neurons",width},{"n_hidden_layers",depth},{"activation","ReLU"},{"output_activation","None"}}));
        weights.resize(net->n_params());grad.resize(net->n_params());net->set_params(weights.data(),weights.data(),grad.data());
        opt.reset(tcnn::create_optimizer<H>({{"otype","Adam"},{"learning_rate",lr},{"beta1",.9f},{"beta2",.999f},
            {"epsilon",1e-15f},{"l2_reg",0.f},{"non_matrix_l2_reg",0.f}}));opt->allocate(net->n_params(),net->layer_sizes());
    }
};
struct Probe {
    cudaStream_t stream;int total,width,height,mode,warp_limit,active=L;float* grid;
    const int* desc;const uint64_t* bits;const int* prefix;const uint8_t* cells;const uint8_t* rgb;
    std::unique_ptr<Head> gate,decoder;
    std::unique_ptr<tcnn::Encoding<H>> encoding;
    std::unique_ptr<tcnn::Optimizer<float>> grid_opt;
    std::unique_ptr<tcnn::Optimizer<H>> half_opt;
    GPU<float> xy,target,squares,grad,sigmoid;
    GPU<H> raw,weighted,logits,output,dy,dz,direct,da,dr,hgrad,grid_half;
    GPU<uint8_t> vis;
    GPU<odie::Level> levels;
    GPU<int> dirty_flags,dirty_list,dirty_count,hash_descriptors;
    std::unique_ptr<tcnn::Context> gate_ctx,decoder_ctx,encoding_ctx;
    // Captured graphs retain pointers into these contexts, including across short eager batches.
    std::vector<std::unique_ptr<tcnn::Context>> captured_contexts;
    Probe(const json& j):stream(reinterpret_cast<cudaStream_t>(j.at("stream").get<uint64_t>())),total(j.at("total")),
        width(j.at("width")),height(j.at("height")),mode(j.at("mode")),warp_limit(j.value("warp_limit",128)),
        grid(ptr<float>(j,"grid")),desc(j.contains("hash_encoding")?nullptr:ptr<int>(j,"descriptors")),
        bits(j.contains("hash_encoding")?nullptr:ptr<uint64_t>(j,"node_bitmap")),
        prefix(j.contains("hash_encoding")?nullptr:ptr<int>(j,"node_prefix")),
        cells(j.contains("hash_encoding")?nullptr:ptr<uint8_t>(j,"bitmap")),rgb(ptr<uint8_t>(j,"rgb")){
        if(j.contains("hash_encoding")){
            if(!(mode&4))throw std::runtime_error("HashGrid requires native half gradients");
            encoding.reset(tcnn::create_encoding<H>(2,j.at("hash_encoding"),16));
            if(encoding->n_params()!=total || encoding->padded_output_width()!=W)
                throw std::runtime_error("TCNN HashGrid allocation differs from resolved layout");
            // Reuse the existing gate kernels with full visibility and uniform features per level.
            std::vector<int> host(L*8,0);int f=j.at("hash_encoding").at("n_features_per_level");
            for(int l=0;l<L;++l){host[8*l+5]=f;host[8*l+6]=l*f;}
            hash_descriptors.resize(host.size());hash_descriptors.copy_from_host(host.data());desc=hash_descriptors.data();
        }
#if ODIE_ROUTER_ENABLED
        gate.reset(new Head(L,32,1,ptr<float>(j,"gate"),j.at("gate_lr")));
#endif
        decoder.reset(new Head(3,64,2,ptr<float>(j,"decoder"),j.at("decoder_lr")));
        xy.resize(2*B);target.resize(3*B);squares.resize(B);grad.resize(total);grad.memset(0);
        raw.resize(W*B);output.resize(16*B);dy.resize(16*B);dz.resize(W*B);vis.resize(L*B);
        if(encoding)vis.memset(1);
#if ODIE_ROUTER_ENABLED
        sigmoid.resize(L*B);weighted.resize(W*B);logits.resize(G*B);
        direct.resize(W*B);da.resize(G*B);dr.resize(W*B);
#endif
        hgrad.resize(total);hgrad.memset(0);grid_half.resize(total);
        if(encoding)encoding->set_params(grid_half.data(),grid_half.data(),hgrad.data());
        int tiles=blocks(total);dirty_flags.resize(tiles);dirty_flags.memset(0);dirty_list.resize(tiles);dirty_count.resize(1);dirty_count.memset(0);
        std::vector<int> host(L*8);CUDA_CHECK_THROW(cudaMemcpy(host.data(),desc,host.size()*4,cudaMemcpyDeviceToHost));
        std::vector<odie::Level> ls;for(int l=0;l<L;++l){const int* d=host.data()+8*l;
            ls.push_back({d[0],d[1],0,d[5],d[6],d[2],0,d[4],d[7],1});}
        levels.resize(L);levels.copy_from_host(ls.data());
        json cfg={{"otype","Adam"},{"learning_rate",j.at("grid_lr")},{"beta1",.9f},{"beta2",.999f},
            {"epsilon",1e-15f},{"l2_reg",0.f},{"non_matrix_l2_reg",0.f},{"skip_zero_grad_non_matrix_params",true}};
        grid_opt.reset(tcnn::create_optimizer<float>(cfg));grid_opt->allocate(total,{});
        half_opt.reset(tcnn::create_optimizer<H>(cfg));half_opt->allocate(total,{});
        sync_weights();CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
    void configure(const json& j){
        active=j.at("active");
        if(active<2 || active>L)throw std::runtime_error("Invalid active prefix");
        if(!encoding){
        desc=ptr<int>(j,"descriptors");bits=ptr<uint64_t>(j,"node_bitmap");
        prefix=ptr<int>(j,"node_prefix");cells=ptr<uint8_t>(j,"bitmap");
        }
        grid_opt->set_learning_rate(j.at("grid_lr"));half_opt->set_learning_rate(j.at("grid_lr"));
        if(gate)gate->opt->set_learning_rate(j.at("gate_lr"));
        decoder->opt->set_learning_rate(j.at("decoder_lr"));
    }
    json state_summary(int begin,int end){
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        json gridstate=(mode&4)?half_opt->serialize():grid_opt->serialize();
        if(begin>=0){
            bool pristine=true;
            for(const char* key:{"first_moments_binary","second_moments_binary","param_steps_binary"}){
                const auto& bytes=gridstate[key].get_binary();
                for(int i=begin*4;i<end*4;++i)if(bytes[i]){pristine=false;break;}
            }
            return {{"pristine",pristine}};
        }
        json states=json::array({gridstate});
        if(gate)states.push_back(gate->opt->serialize());
        states.push_back(decoder->opt->serialize());
        for(auto& state:states){state.erase("current_step");state.erase("base_learning_rate");}
        auto bytes=json::to_msgpack(states);uint64_t hash=14695981039346656037ull;
        for(auto b:bytes){hash^=b;hash*=1099511628211ull;}
        return {{"hash_fnv1a64",std::to_string(hash)},{"bytes",bytes.size()}};
    }
    void sync_weights(){
        if(encoding)odie::cast_float_half<<<blocks(total),256,0,stream>>>(grid,grid_half.data(),total);
        if(gate)odie::cast_float_half<<<blocks(gate->weights.size()),256,0,stream>>>(gate->master,gate->weights.data(),gate->weights.size());
        odie::cast_float_half<<<blocks(decoder->weights.size()),256,0,stream>>>(decoder->master,decoder->weights.data(),decoder->weights.size());
    }
    void step(const int64_t* ids,int real,bool update){
        if(real<=0 || real>B)throw std::runtime_error("Invalid native batch size");
        int n=(real+255)/256*256;
        bool fused=mode&1,half=mode&4,dirty=mode&8,parallel=mode&16;
        input_batch<<<blocks(n),256,0,stream>>>(ids,rgb,xy.data(),target.data(),n,real,width,height,fused);
        Mat r(raw.data(),W,n),out(output.data(),16,n);
        tcnn::GPUMatrixDynamic<float> coords(xy.data(),2,n);
        if(encoding)encoding_ctx=encoding->forward(stream,coords,&r,false,false);
        else query<<<dim3(blocks(n),L),256,0,stream>>>(xy.data(),grid,desc,bits,prefix,cells,raw.data(),vis.data(),n,active);
#if ODIE_ROUTER_ENABLED
        Mat lg(logits.data(),G,n),wt(weighted.data(),W,n);
        gate_ctx=gate->net->forward(stream,r,&lg,false,true);
        if(parallel)parallel_gate<<<blocks(n*L),256,0,stream>>>(raw.data(),logits.data(),vis.data(),desc,weighted.data(),sigmoid.data(),n);
        else odie::gate_weight<<<(n+127)/128,128,0,stream>>>(raw.data(),logits.data(),vis.data(),levels.data(),weighted.data(),n,RAW,W,G,L,ODIE_GATE_K);
#else
        Mat wt(raw.data(),W,n);
#endif
        decoder_ctx=decoder->net->forward(stream,wt,&out,false,true);
        loss_grad<<<blocks(n),256,0,stream>>>(output.data(),rgb,target.data(),ids,dy.data(),squares.data(),n,real,fused);
        Mat d_y(dy.data(),16,n),d_z(dz.data(),W,n);
        decoder->net->backward(stream,*decoder_ctx,wt,out,d_y,&d_z,false,tcnn::GradientMode::Overwrite);
#if ODIE_ROUTER_ENABLED
        Mat d_a(da.data(),G,n),d_r(dr.data(),W,n);
        if(parallel)parallel_gate_back<<<blocks(n*L),256,0,stream>>>(raw.data(),sigmoid.data(),vis.data(),desc,dz.data(),direct.data(),da.data(),n,fused && !encoding);
        else odie::weight_backward<<<(n+127)/128,128,0,stream>>>(raw.data(),logits.data(),dz.data(),vis.data(),levels.data(),direct.data(),da.data(),n,RAW,W,G,L,ODIE_GATE_K,128.f);
        gate->net->backward(stream,*gate_ctx,r,lg,d_a,&d_r,false,tcnn::GradientMode::Overwrite);
#endif
        if(encoding){
            // Restore the grid's single loss scale before official encoding backward.
#if ODIE_ROUTER_ENABLED
            hash_router_gradient<<<blocks(W*n),256,0,stream>>>(direct.data(),dr.data(),dz.data(),W*n);
#else
            odie::half_unscale<<<blocks(W*n),256,0,stream>>>(dz.data(),W*n,128.f);
#endif
            encoding->backward(stream,*encoding_ctx,coords,r,d_z,nullptr,false,tcnn::GradientMode::Overwrite);
        }else{
        if(dirty){
            clear_dirty<<<blocks(total),TILE,0,stream>>>(grad.data(),hgrad.data(),dirty_flags.data(),dirty_list.data(),dirty_count.data(),total);
            CUDA_CHECK_THROW(cudaMemsetAsync(dirty_count.data(),0,4,stream));
        }else{
            CUDA_CHECK_THROW(cudaMemsetAsync(grad.data(),0,grad.n_bytes(),stream));
            if(half)CUDA_CHECK_THROW(cudaMemsetAsync(hgrad.data(),0,hgrad.n_bytes(),stream));
        }
        grid_back<<<dim3(blocks(real),L),256,0,stream>>>(xy.data(),ODIE_ROUTER_ENABLED?(parallel&&fused?dz.data():direct.data()):dz.data(),dr.data(),desc,bits,prefix,cells,grad.data(),hgrad.data(),real,warp_limit,half,
            dirty?dirty_flags.data():nullptr,dirty_list.data(),dirty_count.data(),sigmoid.data(),parallel&&fused);
        if(half)cast_ranges<<<dim3(128,L),256,0,stream>>>(grad.data(),hgrad.data(),desc,total);
        }
        if(!fused){
            if(gate)odie::half_unscale<<<blocks(gate->grad.size()),256,0,stream>>>(gate->grad.data(),gate->grad.size(),128.f);
            odie::half_unscale<<<blocks(decoder->grad.size()),256,0,stream>>>(decoder->grad.data(),decoder->grad.size(),128.f);
        }
        if(update){
            if(half)half_opt->step(stream,128.f,grid,grid_half.data(),hgrad.data());
            else grid_opt->step(stream,128.f,grid,grid,grad.data());
            float scale=fused?16384.f:128.f;
            if(gate)gate->opt->step(stream,scale,gate->master,gate->weights.data(),gate->grad.data());
            decoder->opt->step(stream,scale,decoder->master,decoder->weights.data(),decoder->grad.data());
        }
        CUDA_CHECK_THROW(cudaPeekAtLastError());
    }
    json pointers(){return {{"grid_grad",uint64_t(grad.data())},{"half_grad",uint64_t(hgrad.data())},
        {"gate_grad",gate?uint64_t(gate->grad.data()):uint64_t(0)},{"decoder_grad",uint64_t(decoder->grad.data())},
        {"output",uint64_t(output.data())},{"raw",uint64_t(raw.data())},{"squares",uint64_t(squares.data())},
        {"dirty_count",uint64_t(dirty_count.data())}};}
};
thread_local std::string error,reply;
extern "C" {
const char* odie_error(){return error.c_str();}
void* odie_create(const char* cfg){try{return new Probe(json::parse(cfg));}catch(const std::exception& e){error=e.what();return nullptr;}}
int odie_step(void* p,uint64_t ids,int n,int update){try{static_cast<Probe*>(p)->step(reinterpret_cast<int64_t*>(ids),n,update);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
const char* odie_pointers(void* p){reply=static_cast<Probe*>(p)->pointers().dump();return reply.c_str();}
const char* odie_state(void* p,int begin,int end){try{reply=static_cast<Probe*>(p)->state_summary(begin,end).dump();return reply.c_str();}catch(const std::exception& e){error=e.what();return nullptr;}}
int odie_configure(void* p,const char* cfg){try{static_cast<Probe*>(p)->configure(json::parse(cfg));return 0;}catch(const std::exception& e){error=e.what();return -1;}}
void odie_release_contexts(void* p){static_cast<Probe*>(p)->captured_contexts.clear();}
int odie_sync_weights(void* p){try{static_cast<Probe*>(p)->sync_weights();return 0;}catch(const std::exception& e){error=e.what();return -1;}}
void odie_pin_contexts(void* p){auto* q=static_cast<Probe*>(p);if(q->gate_ctx)q->captured_contexts.emplace_back(std::move(q->gate_ctx));if(q->encoding_ctx)q->captured_contexts.emplace_back(std::move(q->encoding_ctx));q->captured_contexts.emplace_back(std::move(q->decoder_ctx));}
int odie_copy(uint64_t dst,uint64_t src,uint64_t bytes,uint64_t stream){return int(cudaMemcpyAsync(reinterpret_cast<void*>(dst),reinterpret_cast<void*>(src),bytes,cudaMemcpyDeviceToDevice,reinterpret_cast<cudaStream_t>(stream)));}
void odie_destroy(void* p){delete static_cast<Probe*>(p);}
}

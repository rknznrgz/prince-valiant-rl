// Replaces ONLY the copied API entry points; RecorderState::transfer unchanged.
struct Kind18Owner {
    NES::Emulator emulator;
    NES::Kind18Observer observer;
    explicit Kind18Owner(const char* path):emulator(path) {
        observer.prg=NES::RecorderState::prg_base(emulator);
    }
};
extern "C" {
__declspec(dllexport) const char* rec_build(){return "BASE_RECORDER_BUILD_ID";}
__declspec(dllexport) void* rec_create(const char* path){try{return new Kind18Owner(path);}catch(...){return nullptr;}}
__declspec(dllexport) void rec_destroy(Kind18Owner* e){delete e;}
__declspec(dllexport) void rec_reset(Kind18Owner* e){e->observer.clear();e->emulator.reset();}
__declspec(dllexport) unsigned char* rec_ram(Kind18Owner* e){return e->emulator.get_memory_buffer();}
__declspec(dllexport) unsigned int* rec_screen(Kind18Owner* e){return e->emulator.get_screen_buffer();}
__declspec(dllexport) int rec_step(Kind18Owner* e,int mask){
    try{++e->observer.frame;NES::ObserverScope scope(e->observer);
        *e->emulator.get_controller(0)=mask;e->emulator.step();return 0;
    }catch(...){e->observer.stage=NES::Kind18Observer::NONE;return -1;}}
__declspec(dllexport) int rec_save(Kind18Owner* e,unsigned char* out,int capacity){
    try{NES::RecorderState s;s.transfer(e->emulator);if(capacity<int(s.bytes.size()))return -1;
        std::memcpy(out,s.bytes.data(),s.bytes.size());return int(s.bytes.size());}catch(...){return -1;}}
__declspec(dllexport) int rec_load(Kind18Owner* e,const unsigned char* data,int length){
    e->observer.clear();
    try{NES::RecorderState s;s.reading=true;s.bytes.assign(data,data+length);s.transfer(e->emulator);return 0;}catch(...){return -1;}}
__declspec(dllexport) unsigned rec_observer_abi(){return 2;}
// Exposes the exact counter-commit predicate to focused experimental regressions.
__declspec(dllexport) unsigned rec_observer_counter_commit_valid(unsigned before,unsigned after){
    return NES::Kind18Observer::lethal_counter_commit(before,after)?1u:0u;
}
__declspec(dllexport) unsigned rec_observer_overflow(Kind18Owner* e){return e->observer.overflow?1:0;}
// Read consumes an event. Overflow remains sticky until reset/load/capacity change.
__declspec(dllexport) int rec_observer_read(Kind18Owner* e,NES::Kind18Event* out,int capacity){
    if(capacity<0 || (capacity>0 && out==nullptr))return -1;
    if(capacity>0 && e->observer.count){
        *out=e->observer.events[0];
        if(e->observer.count>1)e->observer.events[0]=e->observer.events[1];
        --e->observer.count;return 1;
    }
    return 0;}
// Capacity0 intentionally exercises overflow on the SAME real executed event.
// This changes only observer storage and clears observation state, never game RAM.
__declspec(dllexport) int rec_observer_capacity(Kind18Owner* e,unsigned capacity){
    if(capacity>2)return -1;e->observer.capacity=capacity;e->observer.clear();return 0;}
}

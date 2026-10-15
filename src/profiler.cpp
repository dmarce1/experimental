/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "cosmictiger/profiler.hpp"

#include <hpx/runtime_local/thread_hooks.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <atomic>
#include <bit>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <elf.h>
#include <fcntl.h>
#include <limits>
#include <link.h>
#include <map>
#include <new>
#include <sstream>
#include <stddef.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <ucontext.h>
#endif

namespace profile {
namespace detail {

    enum class Status { dormant, starting, running, stopped };

    class AtomicLock {
        std::atomic_flag busy_ = ATOMIC_FLAG_INIT;

    public:
        void lock() noexcept {
            while (busy_.test_and_set(std::memory_order_acquire)) {
                while (busy_.test(std::memory_order_relaxed)) {
                }
            }
        }
        void unlock() noexcept {
            busy_.clear(std::memory_order_release);
        }
    };

    class AtomicGuard {
        AtomicLock& lock_;

    public:
        explicit AtomicGuard(AtomicLock& lock) noexcept
          : lock_(lock) {
            lock_.lock();
        }
        ~AtomicGuard() {
            lock_.unlock();
        }
        AtomicGuard(AtomicGuard const&) = delete;
        AtomicGuard& operator=(AtomicGuard const&) = delete;
    };

    struct SymbolizedAddress {
        std::string functionName;
        std::string moduleName;
        std::uintptr_t functionAddress;
        std::uintptr_t functionEnd;
        // ELF virtual address: runtime address minus the module's load bias.
        std::uintptr_t moduleOffset;
    };

    struct SymbolExecutableRange {
        std::uintptr_t begin;
        std::uintptr_t end;
    };

    struct SymbolRecord {
        std::uintptr_t begin;
        std::uintptr_t end;
        std::uintptr_t maximumEnd;
        std::string mangledName;
        std::string demangledName;
        bool fullTable;
    };

    struct SymbolModule {
        std::string name;
        std::string path;
        std::uintptr_t loadBias;
        std::vector<SymbolExecutableRange> executableRanges;
        std::vector<SymbolRecord> symbols;
        bool loaded = false;
        std::uint64_t additions = 0;
        std::uint64_t removals = 0;
        bool generationKnown = false;
    };

    class SymbolFile {
    public:
        explicit SymbolFile(std::string const& path) noexcept {
            descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (descriptor_ < 0) {
                return;
            }
            struct stat status{};
            if (::fstat(descriptor_, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0) {
                ::close(descriptor_);
                descriptor_ = -1;
                return;
            }
            bytes_ = static_cast<std::uint64_t>(status.st_size);
        }

        ~SymbolFile() {
            if (descriptor_ >= 0) {
                ::close(descriptor_);
            }
        }

        SymbolFile(SymbolFile const&) = delete;
        SymbolFile& operator=(SymbolFile const&) = delete;

        bool contains(std::uint64_t offset, std::uint64_t bytes) const noexcept {
            return descriptor_ >= 0 && offset <= bytes_ && bytes <= bytes_ - offset;
        }

        bool read(std::uint64_t offset, void* destination, size_t bytes) const noexcept {
            constexpr size_t kMaximumReadChunk = 1024 * 1024;
            constexpr std::uint64_t kMaximumFileOffset = (std::uint64_t{1} << 63) - 1;
            if (!contains(offset, bytes) || offset > kMaximumFileOffset || bytes > kMaximumFileOffset - offset) {
                return false;
            }
            auto* output = static_cast<unsigned char*>(destination);
            size_t completed = 0;
            while (completed < bytes) {
                size_t const remaining = bytes - completed;
                size_t const requested = remaining < kMaximumReadChunk ? remaining : kMaximumReadChunk;
                ssize_t const count = ::pread(descriptor_, output + completed, requested, static_cast<off_t>(offset + completed));
                if (count < 0 && errno == EINTR) {
                    continue;
                }
                if (count <= 0) {
                    return false;
                }
                completed += static_cast<size_t>(count);
            }
            return true;
        }

    private:
        int descriptor_ = -1;
        std::uint64_t bytes_ = 0;
    };

    class Symbolizer {
    public:
        SymbolizedAddress resolve(std::uintptr_t address) {
            SymbolModule* module = currentModule(address);
            if (module == nullptr) {
                return {hexAddress(address), {}, 0, 0, 0};
            }
            if (!module->loaded) {
                loadSymbols(*module);
                module->loaded = true;
            }
            SymbolRecord* symbol = containingSymbol(*module, address);
            std::uintptr_t const offset = address - module->loadBias;
            if (symbol == nullptr) {
                return {hexAddress(address) + " (" + module->name + "+" + hexAddress(offset) + ")", module->name, 0, 0, offset};
            }
            if (symbol->demangledName.empty()) {
                symbol->demangledName = demangle(symbol->mangledName);
            }
            return {symbol->demangledName, module->name, symbol->begin, symbol->end, offset};
        }

    private:
        using ModuleKey = std::pair<std::uintptr_t, std::string>;
        using SymbolBuckets = std::map<std::uintptr_t, std::vector<SymbolRecord>>;

        // Cache by module identity, not by raw PC. The loader is queried on every
        // lookup, so a newly loaded DSO cannot be mistaken for an earlier module.
        std::map<ModuleKey, SymbolModule> modules_;

        static std::string hexAddress(std::uintptr_t value) {
            constexpr char kDigits[] = "0123456789abcdef";
            constexpr size_t kNibbleBits = 4;
            constexpr std::uintptr_t kNibbleMask = 15;
            constexpr size_t kHexDigitsPerByte = 2;
            constexpr size_t kPrefixBytes = 2;
            constexpr size_t kBufferBytes = kPrefixBytes + sizeof(std::uintptr_t) * kHexDigitsPerByte;
            char buffer[kBufferBytes];
            size_t position = kBufferBytes;
            do {
                buffer[--position] = kDigits[value & kNibbleMask];
                value >>= kNibbleBits;
            } while (value != 0);
            buffer[--position] = 'x';
            buffer[--position] = '0';
            return std::string(buffer + position, kBufferBytes - position);
        }

        static std::string demangle(std::string const& name) {
            int status = 0;
            char* const buffer = abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
            if (buffer == nullptr || status != 0) {
                std::free(buffer);
                return name;
            }
            try {
                std::string result(buffer);
                std::free(buffer);
                return result;
            } catch (...) {
                std::free(buffer);
                throw;
            }
        }

        static bool sameRanges(std::vector<SymbolExecutableRange> const& first, std::vector<SymbolExecutableRange> const& second) noexcept {
            if (first.size() != second.size()) {
                return false;
            }
            for (size_t index = 0; index < first.size(); index++) {
                if (first[index].begin != second[index].begin || first[index].end != second[index].end) {
                    return false;
                }
            }
            return true;
        }

        template <typename Visitor>
        static int visitModule(dl_phdr_info* information, size_t informationBytes, void* context) noexcept {
            // The C callback cannot carry an annotated_function object directly;
            // its user callable was annotated before binding this bridge.
            try {
                return (*static_cast<Visitor*>(context))(*information, informationBytes);
            } catch (...) {
                // Never propagate an exception through the C loader callback.
                return 1;
            }
        }

        SymbolModule* currentModule(std::uintptr_t address) {
            constexpr std::uintptr_t kMaximumAddress = ~std::uintptr_t{0};
            constexpr size_t kPathBytes = 4096;
            SymbolModule* selected = nullptr;
            auto visitor = hpx::annotated_function(
                [this, address, &selected](dl_phdr_info const& information, size_t informationBytes) {
                    std::uintptr_t const bias = information.dlpi_addr;
                    bool containsAddress = false;
                    std::vector<SymbolExecutableRange> ranges;
                    for (size_t index = 0; index < information.dlpi_phnum; index++) {
                        auto const& segment = information.dlpi_phdr[index];
                        if (segment.p_type != PT_LOAD || (segment.p_flags & PF_X) == 0 || segment.p_vaddr > kMaximumAddress - bias) {
                            continue;
                        }
                        std::uintptr_t const begin = bias + segment.p_vaddr;
                        if (segment.p_memsz > kMaximumAddress - begin) {
                            continue;
                        }
                        std::uintptr_t const end = begin + segment.p_memsz;
                        ranges.push_back({begin, end});
                        containsAddress = containsAddress || (address >= begin && address < end);
                    }
                    if (!containsAddress || address < bias) {
                        return 0;
                    }
                    bool const generationKnown = informationBytes >= offsetof(dl_phdr_info, dlpi_subs) + sizeof(information.dlpi_subs);
                    std::uint64_t const additions = generationKnown ? information.dlpi_adds : 0;
                    std::uint64_t const removals = generationKnown ? information.dlpi_subs : 0;
                    bool const mainExecutable = information.dlpi_name == nullptr || information.dlpi_name[0] == '\0';
                    std::string const path = mainExecutable ? "/proc/self/exe" : information.dlpi_name;
                    auto const key = std::make_pair(bias, path);
                    auto found = modules_.find(key);
                    if (found == modules_.end()) {
                        std::string name = path;
                        if (mainExecutable) {
                            char buffer[kPathBytes];
                            ssize_t const count = ::readlink(path.c_str(), buffer, sizeof(buffer));
                            if (count > 0 && static_cast<size_t>(count) < sizeof(buffer)) {
                                name.assign(buffer, static_cast<size_t>(count));
                            }
                        }
                        found =
                            modules_.emplace(key, SymbolModule{std::move(name), path, bias, std::move(ranges), {}, false, additions, removals, generationKnown})
                                .first;
                    } else if (!sameRanges(found->second.executableRanges, ranges) || !generationKnown || !found->second.generationKnown ||
                        found->second.additions != additions || found->second.removals != removals) {
                        found->second.executableRanges = std::move(ranges);
                        found->second.symbols.clear();
                        found->second.loaded = false;
                        found->second.additions = additions;
                        found->second.removals = removals;
                        found->second.generationKnown = generationKnown;
                    }
                    selected = &found->second;
                    return 1;
                },
                "profile::Symbolizer find executable module");
            ::dl_iterate_phdr(&visitModule<decltype(visitor)>, &visitor);
            return selected;
        }

        static bool readSection(SymbolFile const& file, Elf64_Ehdr const& header, std::uint64_t index, Elf64_Shdr& section) noexcept {
            constexpr std::uint64_t kMaximumOffset = ~std::uint64_t{0};
            if (header.e_shentsize < sizeof(section) || index > (kMaximumOffset - header.e_shoff) / header.e_shentsize) {
                return false;
            }
            return file.read(header.e_shoff + index * header.e_shentsize, &section, sizeof(section));
        }

        static bool executableSymbol(SymbolModule const& module, std::uintptr_t begin, std::uintptr_t end) noexcept {
            for (auto const& range : module.executableRanges) {
                if (begin >= range.begin && begin < range.end && end <= range.end) {
                    return true;
                }
            }
            return false;
        }

        static void loadSymbols(SymbolModule& module) {
            constexpr std::uintptr_t kMaximumAddress = ~std::uintptr_t{0};
            constexpr std::uint64_t kMaximumSections = 1024 * 1024;
            constexpr std::uint64_t kMaximumSymbolTableBytes = 64 * 1024 * 1024;
            constexpr std::uint64_t kMaximumStringTableBytes = 128 * 1024 * 1024;
            constexpr size_t kMaximumNameBytes = 64 * 1024;
            constexpr size_t kMaximumFunctionSymbols = 1024 * 1024;
            constexpr size_t kMaximumStoredNameBytes = 128 * 1024 * 1024;
            constexpr unsigned char kNativeData = std::endian::native == std::endian::little ? ELFDATA2LSB : ELFDATA2MSB;
            static_assert(sizeof(std::uintptr_t) >= sizeof(Elf64_Addr));
            SymbolFile const file(module.path);
            Elf64_Ehdr header{};
            if (!file.read(0, &header, sizeof(header)) || std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64 ||
                header.e_ident[EI_DATA] != kNativeData || header.e_ident[EI_VERSION] != EV_CURRENT || header.e_version != EV_CURRENT ||
                (header.e_type != ET_EXEC && header.e_type != ET_DYN) || header.e_shoff == 0 || header.e_ehsize < sizeof(header) ||
                header.e_shentsize < sizeof(Elf64_Shdr)) {
                return;
            }
            std::uint64_t sectionCount = header.e_shnum;
            if (sectionCount == 0) {
                Elf64_Shdr first{};
                if (!readSection(file, header, 0, first)) {
                    return;
                }
                sectionCount = first.sh_size;
            }
            if (sectionCount == 0 || sectionCount > kMaximumSections || !file.contains(header.e_shoff, sectionCount * header.e_shentsize)) {
                return;
            }
            SymbolBuckets buckets;
            std::map<size_t, Elf64_Shdr> codeSections;
            size_t symbolCount = 0;
            size_t storedNameBytes = 0;
            for (std::uint64_t sectionIndex = 0; sectionIndex < sectionCount; sectionIndex++) {
                Elf64_Shdr table{};
                if (!readSection(file, header, sectionIndex, table)) {
                    return;
                }
                if ((table.sh_type != SHT_SYMTAB && table.sh_type != SHT_DYNSYM) || table.sh_entsize < sizeof(Elf64_Sym) || table.sh_size == 0 ||
                    table.sh_size % table.sh_entsize != 0 || table.sh_size > kMaximumSymbolTableBytes || table.sh_link >= sectionCount ||
                    !file.contains(table.sh_offset, table.sh_size)) {
                    continue;
                }
                Elf64_Shdr strings{};
                if (!readSection(file, header, table.sh_link, strings) || strings.sh_type != SHT_STRTAB || strings.sh_size == 0 ||
                    strings.sh_size > kMaximumStringTableBytes || !file.contains(strings.sh_offset, strings.sh_size)) {
                    continue;
                }
                std::vector<unsigned char> symbolBytes(static_cast<size_t>(table.sh_size));
                std::vector<char> nameBytes(static_cast<size_t>(strings.sh_size));
                if (!file.read(table.sh_offset, symbolBytes.data(), symbolBytes.size()) || !file.read(strings.sh_offset, nameBytes.data(), nameBytes.size())) {
                    continue;
                }
                size_t const entries = static_cast<size_t>(table.sh_size / table.sh_entsize);
                for (size_t index = 0; index < entries; index++) {
                    Elf64_Sym symbol{};
                    std::memcpy(&symbol, symbolBytes.data() + index * table.sh_entsize, sizeof(symbol));
                    if (ELF64_ST_TYPE(symbol.st_info) != STT_FUNC || symbol.st_size == 0 || symbol.st_shndx == SHN_UNDEF || symbol.st_shndx >= SHN_LORESERVE ||
                        symbol.st_shndx >= sectionCount || symbol.st_name == 0 || symbol.st_name >= nameBytes.size() ||
                        symbol.st_value > kMaximumAddress - module.loadBias) {
                        continue;
                    }
                    auto codeSection = codeSections.find(symbol.st_shndx);
                    if (codeSection == codeSections.end()) {
                        Elf64_Shdr definition{};
                        if (!readSection(file, header, symbol.st_shndx, definition)) {
                            continue;
                        }
                        codeSection = codeSections.emplace(symbol.st_shndx, definition).first;
                    }
                    auto const& definition = codeSection->second;
                    if ((definition.sh_flags & SHF_EXECINSTR) == 0 || (definition.sh_flags & SHF_ALLOC) == 0 || definition.sh_type == SHT_NOBITS ||
                        !file.contains(definition.sh_offset, definition.sh_size) || symbol.st_value < definition.sh_addr ||
                        symbol.st_value - definition.sh_addr > definition.sh_size ||
                        symbol.st_size > definition.sh_size - (symbol.st_value - definition.sh_addr)) {
                        continue;
                    }
                    std::uintptr_t const begin = module.loadBias + symbol.st_value;
                    if (symbol.st_size > kMaximumAddress - begin) {
                        continue;
                    }
                    std::uintptr_t const end = begin + symbol.st_size;
                    if (!executableSymbol(module, begin, end)) {
                        continue;
                    }
                    char const* name = nameBytes.data() + symbol.st_name;
                    size_t const available = nameBytes.size() - symbol.st_name;
                    size_t const limit = available < kMaximumNameBytes ? available : kMaximumNameBytes;
                    auto const* terminator = static_cast<char const*>(std::memchr(name, '\0', limit));
                    if (terminator == nullptr || terminator == name) {
                        continue;
                    }
                    size_t const length = static_cast<size_t>(terminator - name);
                    if (symbolCount >= kMaximumFunctionSymbols || length > kMaximumStoredNameBytes - storedNameBytes) {
                        continue;
                    }
                    bool duplicate = false;
                    auto& bucket = buckets[begin];
                    for (auto& entry : bucket) {
                        if (entry.end == end && entry.mangledName.compare(0, entry.mangledName.size(), name, length) == 0) {
                            entry.fullTable = entry.fullTable || table.sh_type == SHT_SYMTAB;
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        bucket.push_back({begin, end, 0, std::string(name, length), {}, table.sh_type == SHT_SYMTAB});
                        symbolCount++;
                        storedNameBytes += length;
                    }
                }
            }
            module.symbols.reserve(symbolCount);
            std::uintptr_t maximumEnd = 0;
            for (auto& bucket : buckets) {
                for (auto& record : bucket.second) {
                    maximumEnd = maximumEnd < record.end ? record.end : maximumEnd;
                    record.maximumEnd = maximumEnd;
                    module.symbols.push_back(std::move(record));
                }
            }
        }

        static SymbolRecord* containingSymbol(SymbolModule& module, std::uintptr_t address) noexcept {
            size_t lower = 0;
            size_t upper = module.symbols.size();
            while (lower < upper) {
                size_t const middle = lower + (upper - lower) / 2;
                if (module.symbols[middle].begin <= address) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            SymbolRecord* best = nullptr;
            while (lower != 0) {
                lower--;
                auto& candidate = module.symbols[lower];
                if (candidate.maximumEnd <= address) {
                    break;
                }
                if (address >= candidate.begin && address < candidate.end) {
                    if (best == nullptr || candidate.end - candidate.begin < best->end - best->begin ||
                        (candidate.end - candidate.begin == best->end - best->begin && candidate.fullTable && !best->fullTable) ||
                        (candidate.end - candidate.begin == best->end - best->begin && candidate.fullTable == best->fullTable &&
                            candidate.mangledName < best->mangledName)) {
                        best = &candidate;
                    }
                }
            }
            return best;
        }
    };

    struct Bucket {
        std::atomic<std::uintptr_t> address{};
        std::atomic<std::uint64_t> count{};
    };

    class SampleBuffer {
        static size_t constexpr kBucketCount_ = 16384;
        static size_t constexpr kMaximumProbes_ = 8;
        static unsigned constexpr kHashShift_ = 4;
        static std::uintptr_t constexpr kHashMultiplier_ = 11400714819323198485ULL;
        Bucket buckets_[kBucketCount_];
        static std::uint64_t constexpr kEnabledBit_ = std::uint64_t{1} << 63;
        static std::uint64_t constexpr kStoppedBit_ = std::uint64_t{1} << 62;
        static std::uint64_t constexpr kActiveMask_ = kStoppedBit_ - 1;
        std::atomic<std::uint64_t> samplingGate_{};
        std::atomic<bool> timerActive_{};
        std::atomic<std::uint64_t> samples_{};
        std::atomic<std::uint64_t> dropped_{};
        std::atomic<std::uint64_t> overruns_{};
        std::uint64_t intervalNanoseconds_{};
        std::string threadName_;
        SampleBuffer* next_{};
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
        timer_t timer_{};
#endif

    public:
        SampleBuffer(std::uint64_t intervalNanoseconds, std::string threadName)
          : intervalNanoseconds_(intervalNanoseconds)
          , threadName_(std::move(threadName)) {}
        void setNext(SampleBuffer* next) noexcept {
            next_ = next;
        }
        SampleBuffer* next() const noexcept {
            return next_;
        }
        void enable() noexcept {
            std::uint64_t expected = 0;
            samplingGate_.compare_exchange_strong(expected, kEnabledBit_, std::memory_order_acq_rel);
        }
        void disable() noexcept {
            // Closing the gate is permanent, including if stop races initial setup.
            // The handler never waits; ordinary teardown drains the admitted writer.
            samplingGate_.fetch_or(kStoppedBit_, std::memory_order_acq_rel);
            while ((samplingGate_.load(std::memory_order_acquire) & kActiveMask_) != 0) {
            }
        }
        // This entire path is signal-safe: no TLS, library containers, allocation,
        // locks, HPX, floating-point work, symbolization, or unbounded traversal.
        void sample(std::uintptr_t address, int overruns) noexcept {
            std::uint64_t expected = kEnabledBit_;
            if (!samplingGate_.compare_exchange_strong(expected, kEnabledBit_ + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return;
            }
            samples_.fetch_add(1, std::memory_order_relaxed);
            if (overruns > 0) {
                overruns_.fetch_add(static_cast<std::uint64_t>(overruns), std::memory_order_relaxed);
            }
            if (address != 0) {
                size_t const first = ((address >> kHashShift_) * kHashMultiplier_) & (kBucketCount_ - 1);
                for (size_t probe = 0; probe < kMaximumProbes_; probe++) {
                    auto& bucket = buckets_[(first + probe) & (kBucketCount_ - 1)];
                    auto const savedAddress = bucket.address.load(std::memory_order_relaxed);
                    if (savedAddress == 0) {
                        // Each cookie has one targeted OS worker, and the sampling
                        // signal is blocked during its own handler: one writer.
                        bucket.address.store(address, std::memory_order_release);
                        bucket.count.fetch_add(1, std::memory_order_relaxed);
                        samplingGate_.fetch_sub(1, std::memory_order_release);
                        return;
                    }
                    if (savedAddress == address) {
                        bucket.count.fetch_add(1, std::memory_order_relaxed);
                        samplingGate_.fetch_sub(1, std::memory_order_release);
                        return;
                    }
                }
            }
            dropped_.fetch_add(1, std::memory_order_relaxed);
            samplingGate_.fetch_sub(1, std::memory_order_release);
        }
        static constexpr size_t bucketCount() noexcept {
            return kBucketCount_;
        }
        Bucket const& bucket(size_t index) const noexcept {
            return buckets_[index];
        }
        std::uint64_t samples() const noexcept {
            return samples_.load(std::memory_order_relaxed);
        }
        std::uint64_t dropped() const noexcept {
            return dropped_.load(std::memory_order_relaxed);
        }
        std::uint64_t overruns() const noexcept {
            return overruns_.load(std::memory_order_relaxed);
        }
        std::uint64_t intervalNanoseconds() const noexcept {
            return intervalNanoseconds_;
        }
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
        void setTimer(timer_t timer) noexcept {
            timer_ = timer;
            timerActive_.store(true, std::memory_order_release);
        }
        timer_t timer() const noexcept {
            return timer_;
        }
        void stopTimer() noexcept {
            disable();
            if (timerActive_.exchange(false, std::memory_order_acq_rel)) {
                itimerspec disarmed{};
                ::timer_settime(timer_, 0, &disarmed, nullptr);
                ::timer_delete(timer_);
            }
        }
#else
        void stopTimer() noexcept {
            disable();
        }
#endif
    };

    static_assert(std::atomic<std::uintptr_t>::is_always_lock_free);
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    static_assert(std::atomic<bool>::is_always_lock_free);

    class Registry {
        std::atomic<Status> status_{Status::dormant};
        std::atomic<SampleBuffer*> first_{};
        std::atomic<std::uint64_t> setupFailures_{};
        std::uint64_t intervalNanoseconds_{};
        int signal_{};
        AtomicLock metadataLock_;
        std::string filename_;
        std::map<std::uintptr_t, std::string> registrations_;

    public:
        Status status() const noexcept {
            return status_.load(std::memory_order_acquire);
        }
        bool beginStart() noexcept {
            Status expected = Status::dormant;
            return status_.compare_exchange_strong(expected, Status::starting, std::memory_order_acq_rel);
        }
        bool finishStart() noexcept {
            Status expected = Status::starting;
            return status_.compare_exchange_strong(expected, Status::running, std::memory_order_acq_rel);
        }
        void configure(std::uint64_t intervalNanoseconds, std::string filename, int signal) {
            AtomicGuard guard(metadataLock_);
            intervalNanoseconds_ = intervalNanoseconds;
            filename_ = std::move(filename);
            signal_ = signal;
        }
        std::uint64_t intervalNanoseconds() const noexcept {
            return intervalNanoseconds_;
        }
        int signal() const noexcept {
            return signal_;
        }
        void add(SampleBuffer* buffer) noexcept {
            auto* first = first_.load(std::memory_order_relaxed);
            do {
                buffer->setNext(first);
            } while (!first_.compare_exchange_weak(first, buffer, std::memory_order_release, std::memory_order_relaxed));
        }
        SampleBuffer* first() const noexcept {
            return first_.load(std::memory_order_acquire);
        }
        void stop() noexcept {
            if (status() == Status::dormant) {
                return;
            }
            status_.store(Status::stopped, std::memory_order_release);
            for (auto* buffer = first(); buffer != nullptr; buffer = buffer->next()) {
                buffer->stopTimer();
            }
        }
        void failSetup() noexcept {
            setupFailures_.fetch_add(1, std::memory_order_relaxed);
        }
        std::uint64_t setupFailures() const noexcept {
            return setupFailures_.load(std::memory_order_relaxed);
        }
        void registerFunction(std::uintptr_t address, std::string name) {
            AtomicGuard guard(metadataLock_);
            registrations_.try_emplace(address, std::move(name));
        }
        size_t registrationCount() {
            AtomicGuard guard(metadataLock_);
            return registrations_.size();
        }
        std::string filename() {
            AtomicGuard guard(metadataLock_);
            return filename_;
        }
    };

    Registry& registry() {
        // Records and configuration deliberately survive TLS destruction, pending
        // timer signals, HPX shutdown, and the final atexit report.
        static Registry* state = new Registry;
        return *state;
    }

#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))

    // sigaction requires this C ABI callback. Annotating it would call HPX from a
    // signal handler and violate async-signal safety; only normal hooks are wrapped.
    void sampleHandler(int, siginfo_t* information, void* context) noexcept {
        if (information == nullptr || information->si_code != SI_TIMER || information->si_value.sival_ptr == nullptr || context == nullptr) {
            return;
        }
        auto* buffer = static_cast<SampleBuffer*>(information->si_value.sival_ptr);
        auto const* interrupted = static_cast<ucontext_t const*>(context);
#if defined(__x86_64__)
        std::uintptr_t const address = static_cast<std::uintptr_t>(interrupted->uc_mcontext.gregs[REG_RIP]);
#else
        std::uintptr_t const address = static_cast<std::uintptr_t>(interrupted->uc_mcontext.pc);
#endif
        buffer->sample(address, information->si_overrun);
    }

    int installSignal() {
        for (int candidate = SIGRTMIN; candidate <= SIGRTMAX; candidate++) {
            struct sigaction previous{};
            if (::sigaction(candidate, nullptr, &previous) != 0 || previous.sa_handler != SIG_DFL) {
                continue;
            }
            struct sigaction action{};
            action.sa_sigaction = &sampleHandler;
            action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
            ::sigemptyset(&action.sa_mask);
            if (::sigaction(candidate, &action, nullptr) != 0) {
                continue;
            }
            return candidate;
        }
        throw std::runtime_error("sampling profiler has no unused real-time signal");
    }

    class ThreadSampler {
        static size_t constexpr kMinimumStackSize_ = 65536;
        static std::uint64_t constexpr kNanosecondsPerSecond_ = 1000000000;
        SampleBuffer* buffer_{};
        void* stackMemory_{};
        stack_t previousStack_{};
        sigset_t previousMask_{};
        bool maskSaved_{};
        bool stackReplaced_{};
        int signal_{};

    public:
        ~ThreadSampler() {
            detach();
        }
        void detach() noexcept {
            if (!maskSaved_) {
                return;
            }
            sigset_t mask{};
            ::sigemptyset(&mask);
            ::sigaddset(&mask, signal_);
            ::pthread_sigmask(SIG_BLOCK, &mask, nullptr);
            if (buffer_ != nullptr) {
                buffer_->stopTimer();
            }
            if (stackReplaced_) {
                // If restoration fails, preserve the memory rather than leave the
                // kernel pointing to a freed alternate stack.
                if (::sigaltstack(&previousStack_, nullptr) == 0) {
                    std::free(stackMemory_);
                } else {
                    registry().failSetup();
                }
            }
            stackMemory_ = nullptr;
            stackReplaced_ = false;
            buffer_ = nullptr;
            maskSaved_ = false;
            // A queued notification may still be delivered here; its cookie has
            // process lifetime and is disabled, and the handler remains installed.
            ::pthread_sigmask(SIG_SETMASK, &previousMask_, nullptr);
        }
        bool attach(std::string const& threadName) noexcept {
            if (buffer_ != nullptr) {
                return true;
            }
            auto& state = registry();
            if (state.status() != Status::running) {
                return false;
            }
            try {
                signal_ = state.signal();
                sigset_t mask{};
                ::sigemptyset(&mask);
                ::sigaddset(&mask, signal_);
                if (::pthread_sigmask(SIG_BLOCK, &mask, &previousMask_) != 0) {
                    throw std::runtime_error("unable to block sampling signal during setup");
                }
                maskSaved_ = true;
                if (::sigaltstack(nullptr, &previousStack_) != 0) {
                    throw std::runtime_error("unable to inspect alternate signal stack");
                }
                if ((previousStack_.ss_flags & SS_DISABLE) != 0) {
                    size_t const suggestedSize = static_cast<size_t>(SIGSTKSZ);
                    size_t const stackSize = suggestedSize > kMinimumStackSize_ ? suggestedSize : kMinimumStackSize_;
                    stackMemory_ = std::malloc(stackSize);
                    if (stackMemory_ == nullptr) {
                        throw std::bad_alloc();
                    }
                    stack_t stack{};
                    stack.ss_sp = stackMemory_;
                    stack.ss_size = stackSize;
                    if (::sigaltstack(&stack, nullptr) != 0) {
                        std::free(stackMemory_);
                        stackMemory_ = nullptr;
                        throw std::runtime_error("unable to establish alternate signal stack");
                    }
                    stackReplaced_ = true;
                }
                auto* buffer = new SampleBuffer(state.intervalNanoseconds(), threadName);
                sigevent event{};
                event.sigev_notify = SIGEV_THREAD_ID;
                event.sigev_signo = signal_;
                event.sigev_value.sival_ptr = buffer;
                event._sigev_un._tid = ::gettid();
                timer_t timer{};
                if (::timer_create(CLOCK_THREAD_CPUTIME_ID, &event, &timer) != 0) {
                    delete buffer;
                    throw std::runtime_error("unable to create per-thread CPU sampling timer");
                }
                buffer->setTimer(timer);
                state.add(buffer);
                buffer_ = buffer;
                if (state.status() != Status::running) {
                    detach();
                    return false;
                }
                buffer->enable();
                itimerspec period{};
                period.it_interval.tv_sec = static_cast<time_t>(state.intervalNanoseconds() / kNanosecondsPerSecond_);
                period.it_interval.tv_nsec = static_cast<long>(state.intervalNanoseconds() % kNanosecondsPerSecond_);
                period.it_value = period.it_interval;
                if (state.status() != Status::running || ::timer_settime(timer, 0, &period, nullptr) != 0) {
                    if (state.status() == Status::running) {
                        state.failSetup();
                    }
                    detach();
                    return false;
                }
                if (::pthread_sigmask(SIG_UNBLOCK, &mask, nullptr) != 0) {
                    throw std::runtime_error("unable to unblock sampling signal");
                }
                return true;
            } catch (...) {
                state.failSetup();
                detach();
                return false;
            }
        }
    };

    ThreadSampler& threadSampler() {
        // Normal setup/teardown is genuinely OS-thread-local. The signal handler
        // never reads this object or performs TLS lookup/initialization.
        static thread_local ThreadSampler sampler;
        return sampler;
    }

    void attachCurrentThread(char const* name) noexcept {
        try {
            threadSampler().attach(name == nullptr ? "OS worker" : name);
        } catch (...) {
            registry().failSetup();
        }
    }

    void installThreadHooks() {
        auto previousStart = hpx::get_thread_on_start_func();
        auto previousStop = hpx::get_thread_on_stop_func();
        hpx::register_thread_on_start_func(hpx::annotated_function(
            [previous = std::move(previousStart)](size_t local, size_t global, char const* pool, char const* postfix) mutable {
                if (previous) {
                    previous(local, global, pool, postfix);
                }
                attachCurrentThread(pool);
            },
            "Profiler::startOsWorker"));
        hpx::register_thread_on_stop_func(hpx::annotated_function(
            [previous = std::move(previousStop)](size_t local, size_t global, char const* pool, char const* postfix) mutable {
                threadSampler().detach();
                if (previous) {
                    previous(local, global, pool, postfix);
                }
            },
            "Profiler::stopOsWorker"));
    }

#else

    void attachCurrentThread(char const*) noexcept {}

#endif

    void reportAtExit() noexcept {
        try {
            profileStop();
            auto callback = hpx::annotated_function([] { writeProfileReport(); }, "Profiler::exitReport");
            callback();
        } catch (...) {
            std::fputs("CosmicTiger profiler: exit report failed\n", stderr);
        }
    }

    struct ReportTotals {
        std::uint64_t samples{};
        double seconds{};
    };

}    // namespace detail

bool profileStart(std::uint64_t intervalNanoseconds, std::string const& filename) noexcept {
    std::uint64_t constexpr kNanosecondsPerSecond = 1000000000;
    size_t constexpr kFilenameBufferSize = 80;

    try {
        auto& state = detail::registry();
        if (state.status() == detail::Status::running) {
            detail::attachCurrentThread("calling thread");
            return true;
        }
        if (!state.beginStart()) {
            return false;
        }
        if (intervalNanoseconds == 0 || intervalNanoseconds / kNanosecondsPerSecond > static_cast<std::uint64_t>(std::numeric_limits<time_t>::max())) {
            throw std::invalid_argument("invalid CPU sampling interval");
        }
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
        std::string outputFilename = filename;
        if (outputFilename.empty()) {
            char buffer[kFilenameBufferSize];
            std::snprintf(buffer, sizeof(buffer), "cosmictiger.profile.%ld.txt", static_cast<long>(::getpid()));
            outputFilename = buffer;
        }
        int const signal = detail::installSignal();
        state.configure(intervalNanoseconds, std::move(outputFilename), signal);
        // HPX hook registration happens outside every metadata lock.
        detail::installThreadHooks();
        if (std::atexit(&detail::reportAtExit) != 0) {
            throw std::runtime_error("unable to register profiler exit report");
        }
        if (!state.finishStart()) {
            return false;
        }
        detail::attachCurrentThread("calling thread");
        if (state.setupFailures() != 0) {
            profileStop();
            return false;
        }
        return true;
#else
        (void) filename;
        throw std::runtime_error("CPU sampling profiler requires Linux x86-64 or AArch64");
#endif
    } catch (std::exception const& error) {
        try {
            detail::registry().failSetup();
            detail::registry().stop();
        } catch (...) {
        }
        std::fprintf(stderr, "CosmicTiger profiler: %s\n", error.what());
        return false;
    } catch (...) {
        try {
            detail::registry().failSetup();
            detail::registry().stop();
        } catch (...) {
        }
        std::fputs("CosmicTiger profiler: unable to start sampling\n", stderr);
        return false;
    }
}

void profileStop() noexcept {
    try {
        detail::registry().stop();
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
        detail::threadSampler().detach();
#endif
    } catch (...) {
        std::fputs("CosmicTiger profiler: unable to stop sampling\n", stderr);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
bool profileRegisterFunction(char const* functionName) noexcept {
    try {
        if (functionName == nullptr || functionName[0] == '\0') {
            return false;
        }
        std::uintptr_t address{};
#if defined(__GNUC__) || defined(__clang__)
        address = reinterpret_cast<std::uintptr_t>(__builtin_extract_return_addr(__builtin_return_address(0)));
#endif
        auto& state = detail::registry();
        if (state.status() == detail::Status::dormant && !profileStart()) {
            return false;
        }
        if (state.status() != detail::Status::running) {
            return false;
        }
        detail::attachCurrentThread("registered function caller");
        // Metadata is registered once. Sampling attribution always uses the
        // interrupted PC's actual symbol, never a stale current-function name.
        state.registerFunction(address, functionName);
        return true;
    } catch (...) {
        try {
            detail::registry().failSetup();
        } catch (...) {
        }
        return false;
    }
}

std::string profileReport() {
    double constexpr kSecondsPerNanosecond = 1.0e-9;
    double constexpr kPercentScale = 100.0;
    size_t constexpr kNumberBufferSize = 120;

    auto& state = detail::registry();
    detail::Symbolizer symbolizer;
    using Name = std::pair<std::string, std::string>;
    std::map<Name, detail::ReportTotals> entries;
    std::uint64_t samples = 0;
    std::uint64_t dropped = 0;
    std::uint64_t overruns = 0;
    size_t workers = 0;
    double sampledSeconds = 0.0;
    for (auto const* buffer = state.first(); buffer != nullptr; buffer = buffer->next()) {
        workers++;
        samples += buffer->samples();
        dropped += buffer->dropped();
        overruns += buffer->overruns();
        double const secondsPerSample = static_cast<double>(buffer->intervalNanoseconds()) * kSecondsPerNanosecond;
        sampledSeconds += static_cast<double>(buffer->samples()) * secondsPerSample;
        for (size_t index = 0; index < detail::SampleBuffer::bucketCount(); index++) {
            auto const& bucket = buffer->bucket(index);
            auto const address = bucket.address.load(std::memory_order_acquire);
            auto const count = bucket.count.load(std::memory_order_relaxed);
            if (address == 0 || count == 0) {
                continue;
            }
            auto symbol = symbolizer.resolve(address);
            if (symbol.functionName.empty()) {
                char offset[kNumberBufferSize];
                std::snprintf(offset, sizeof(offset), "<unresolved PC 0x%" PRIxPTR ">", address);
                symbol.functionName = offset;
            }
            auto& entry = entries[{std::move(symbol.functionName), std::move(symbol.moduleName)}];
            entry.samples += count;
            entry.seconds += static_cast<double>(count) * secondsPerSample;
        }
    }
    std::map<double, std::vector<std::pair<Name, detail::ReportTotals>>> sorted;
    for (auto const& entry : entries) {
        sorted[entry.second.seconds].push_back(entry);
    }
    std::ostringstream report;
    report << "CosmicTiger statistical CPU profile (local process, all attached OS workers)\n";
    report << "Exclusive CPU estimates = delivered samples x CPU sampling period; parallel totals can exceed wall time.\n";
    report << "No exact call counts, inclusive time, waiting time, or GPU time are measured.\n";
    report << "     Samples   Est. CPU (s)    CPU %  Function [Module]\n";
    for (auto bucket = sorted.rbegin(); bucket != sorted.rend(); bucket++) {
        for (auto const& [name, entry] : bucket->second) {
            char numbers[kNumberBufferSize];
            double const percentage = sampledSeconds > 0.0 ? kPercentScale * entry.seconds / sampledSeconds : 0.0;
            std::snprintf(numbers, sizeof(numbers), "%12" PRIu64 " %14.6e %8.2f  ", entry.samples, entry.seconds, percentage);
            report << numbers << name.first;
            if (!name.second.empty()) {
                report << " [" << name.second << ']';
            }
            report << '\n';
        }
    }
    report << "Attached OS workers: " << workers << "; registered function labels: " << state.registrationCount() << '\n';
    report << "Delivered samples: " << samples << "; dropped samples: " << dropped << "; timer overruns (missing ticks): " << overruns
           << "; setup failures: " << state.setupFailures() << '\n';
    if (state.status() == detail::Status::running) {
        report << "Sampling is still active; call profileStop() for a final snapshot.\n";
    }
    return report.str();
}

bool writeProfileReport(std::string const& filename) noexcept {
    try {
        auto& state = detail::registry();
        auto const report = profileReport();
        std::string const outputFilename = filename.empty() ? state.filename() : filename;
        bool const terminalWritten = std::fwrite(report.data(), 1, report.size(), stdout) == report.size() && std::fflush(stdout) == 0;
        if (!terminalWritten) {
            std::fputs("CosmicTiger profiler: unable to write terminal report\n", stderr);
        }
        FILE* file = std::fopen(outputFilename.c_str(), "wb");
        if (file == nullptr) {
            std::fprintf(stderr, "CosmicTiger profiler: unable to open report file '%s'\n", outputFilename.c_str());
            return false;
        }
        bool const fileWritten = std::fwrite(report.data(), 1, report.size(), file) == report.size();
        bool const fileClosed = std::fclose(file) == 0;
        if (!fileWritten || !fileClosed) {
            std::fprintf(stderr, "CosmicTiger profiler: unable to finish report file '%s'\n", outputFilename.c_str());
        }
        return terminalWritten && fileWritten && fileClosed;
    } catch (...) {
        std::fputs("CosmicTiger profiler: unable to construct report\n", stderr);
        return false;
    }
}

}    // namespace profile

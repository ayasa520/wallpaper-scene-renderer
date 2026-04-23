#pragma once
#include <memory>
#include <vector>
#include "IBinaryStream.h"

#include "Core/Literals.hpp"

namespace wallpaper
{
namespace fs
{

class LimitedBinaryStream : public IBinaryStream {
public:
    LimitedBinaryStream(std::shared_ptr<IBinaryStream> infs, idx start, isize size)
        : m_pos(0), m_start(start), m_end(start + size), m_infs(infs) {}
    virtual ~LimitedBinaryStream() = default;

private:
    bool CanSeekTo(idx pos) const {
        // A limited stream represents an embedded file inside scene.pkg, and decoders treat it
        // exactly like a normal seekable file.  Seeking to byte 0 is required during codec probing
        // and reset, while seeking to Size() is the normal EOF position used to measure streams.
        // Rejecting either boundary makes packaged audio files fail even though the same unpacked
        // file works through CBinaryStream.
        return pos >= 0 && pos <= Size();
    }

    bool SeekInMPos(void) { return m_infs->SeekSet(m_start + m_pos); }
    bool SeekInPos(idx pos) {
        if (CanSeekTo(pos)) {
            m_pos = pos;
            return SeekInMPos();
        }
        return false;
    }
    bool End() const { return m_pos < 0 || m_pos == Size(); };

protected:
    virtual usize Write_impl(const void*, usize) { return 0; }

public:
    virtual usize Read(void* buffer, usize sizeInByte) {
        if (End() || sizeInByte == 0) return 0;

        const isize available = Size() - m_pos;
        if (available <= 0) return 0;

        isize isizeInByte = (isize)sizeInByte;
        if (isizeInByte > available) isizeInByte = available;
        SeekInMPos();
        const usize read = m_infs->Read(buffer, (usize)isizeInByte);
        // Advance by the bytes actually delivered by the backing package stream.  Codec probes can
        // issue short reads near EOF, and keeping the cursor aligned with the real result makes the
        // limited stream follow ordinary file semantics instead of drifting past the embedded file.
        m_pos += static_cast<isize>(read);
        return read;
    }
    virtual char* Gets(char* buffer, usize sizeStr) {
        Read(buffer, sizeStr);
        return buffer;
    }
    virtual idx  Tell() const { return m_pos; }
    virtual bool SeekSet(idx offset) {
        idx pos = offset;
        return SeekInPos(pos);
    }
    virtual bool SeekCur(idx offset) {
        idx pos = m_pos + offset;
        return SeekInPos(pos);
    }
    virtual bool SeekEnd(idx offset) {
        // Match CBinaryStream/fseek semantics: offset is relative to one-past-the-end, so offset 0
        // lands at EOF and negative offsets move backward from EOF.  Codec libraries commonly use
        // this to discover the length of packaged audio streams before rewinding to byte 0.
        idx pos = Size() + offset;
        return SeekInPos(pos);
    }
    virtual isize Size() const { return m_end - m_start; }

private:
    idx                            m_pos; // File-relative cursor; byte 0 and one-past-EOF are valid seek targets.
    const idx                      m_start;
    const idx                      m_end; // end if m_pos == m_end - m_start
    std::shared_ptr<IBinaryStream> m_infs;
};

} // namespace fs
} // namespace wallpaper

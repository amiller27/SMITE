# SMITE

A rewrite of the METIS graph partitioning library in Rust.  Intended for matrix elimination ordering use, so only implementing the components of METIS relevant for that, and not the other stuff.

Goals:

- Implement the sparse matrix elimination ordering use case of METIS, not the rest of it.  Primarily I'm attempting to replicate the use of METIS by MetisSupport in Eigen.
- Be capable of producing identical results to METIS.  This is likely only desirable in a test mode, since METIS does some things in ways that are more easily written differently in a language that isn't C.  METIS also makes use of an RNG, so matching results requires mocking an RNG with the same random stream METIS used for a given execution.
- Be way more readable than METIS, for educational purposes.  Mostly so I can learn how METIS works, but maybe it'll be useful for others.
- Speed.  It'll be cool if this isn't terribly slow.  It'd be even cooler if it were faster than METIS, although it's possible it'd be easier to do that in C++.  Depending on how fast this turns out, maybe I'll do a rewrite in C++ at some point.
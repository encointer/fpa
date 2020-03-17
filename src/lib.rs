//! Fixed point arithmetic
//!
//! A fixed point number is an alternative representation for a real number.
//! IEEE floats, `f32` and `f64`, being the standard format in processors with
//! Floating Point Units (FPU). You should consider using fixed numbers on
//! systems where there's no FPU and performance is critical as fixed point
//! arithmetic is faster than software emulated IEEE float arithmetic. Do note
//! that fixed point numbers tend to be more prone to overflows as they operate
//! in ranges much smaller than floats.
//!
//! The fixed point numbers exposed in this library use the following naming
//! convention: `IxFy`, where `x` is the number of bits used for the integer
//! part and `y` is the number of bits used for the fractional part.
//!
//! Unlike IEEE floats, fixed points numbers have *fixed* precision. One can
//! exchange range for precision by selecting different values for `x` and `y`:
//!
//! - Range: `[-2 ^ (x - 1), 2 ^ (x - 1) - 2 ^ (-y)]`
//! - Precision: `2 ^ (-y)`
//!
//! For example, the type `I1F7` has range `[-1, 0.9921875]` and precision
//! `0.0078125`.
//!
//! # Examples
//!
//! - Casts
//!
//! ```
//! // https://crates.io/crates/cast
//! extern crate cast;
//! extern crate fpa;
//!
//! use cast::f64;
//! // 32-bit fixed point number, 16 bits for the integer part and 16 bits for
//! // the fractional part
//! use fpa::I16F16;
//!
//! fn main() {
//!     // casts an integer into a fixed point number (infallible)
//!     let q = I16F16(1i8);
//!
//!     // casts the fixed point number into a float (infallible)
//!     let f = f64(q);
//!
//!     assert_eq!(f, 1.);
//! }
//! ```
//!
//! - Arithmetic
//!
//! ```
//! use fpa::I16F16;
//!
//! // NOTE the `f64` -> `I16F16` cast is fallible because of NaN and infinity
//! assert_eq!(I16F16(1.25_f64).unwrap() + I16F16(2.75_f64).unwrap(),
//!            I16F16(4_f64).unwrap());
//!
//! assert_eq!(I16F16(2_f64).unwrap() / I16F16(0.5_f64).unwrap(),
//!            I16F16(4_f64).unwrap());
//!
//! assert_eq!(I16F16(2_f64).unwrap() * I16F16(0.5_f64).unwrap(),
//!            I16F16(1_f64).unwrap());
//! ```
//!
//! - Trigonometry
//!
//! ```
//! extern crate cast;
//! extern crate fpa;
//!
//! use cast::f64;
//! use fpa::I2F30;
//!
//! fn main() {
//!     let (r, _) = I2F30(0.3_f64).unwrap().polar(I2F30(0.4_f64).unwrap());
//!     assert!((f64(r) - 0.5).abs() < 1e-5);
//! }
//! ```
//!

#![cfg_attr(feature = "const-fn", feature(const_fn))]
#![cfg_attr(feature = "try-from", feature(try_from))]
#![deny(missing_docs)]
//#![deny(warnings)]
#![no_std]

extern crate cast;
extern crate typenum;
extern crate num_traits;

use core::cmp::Ordering;
use core::marker::PhantomData;
use core::{fmt, ops};

#[cfg(feature = "try_from")]
use core::convert::{TryFrom, Infallible};

use cast::{From as CastFrom, Error, f32, f64, i16, i32, i64, i8, isize, u16, u32, u64, u8, usize};
use typenum::{Cmp, Greater, Less, U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10,
              U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23,
              U24, U25, U26, U27, U28, U29, U30, U31, U32, Unsigned};

mod num_traits_impl;

use num_traits::{Zero, One};

/// Fixed point number
///
/// - `BITS` is the integer primitive used to stored the number
/// - `FRAC` is the number of bits used for the fractional part of the number
pub struct Q<BITS, FRAC> {
    bits: BITS,
    _marker: PhantomData<FRAC>,
}

impl<BITS, FRAC> Clone for Q<BITS, FRAC>
where
    BITS: Clone,
{
    fn clone(&self) -> Self {
        Q {
            bits: self.bits.clone(),
            _marker: PhantomData,
        }
    }
}

impl<BITS, FRAC> Copy for Q<BITS, FRAC>
where
    BITS: Copy,
{
}

impl<BITS, FRAC> Eq for Q<BITS, FRAC>
where
    BITS: Eq,
{
}

impl<BITS, FRAC> Ord for Q<BITS, FRAC>
where
    BITS: Ord,
{
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.bits.cmp(&rhs.bits)
    }
}

impl<BITS, FRAC> PartialEq for Q<BITS, FRAC>
where
    BITS: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.bits.eq(&rhs.bits)
    }
}

impl<BITS, FRAC> PartialOrd for Q<BITS, FRAC>
where
    BITS: PartialOrd,
{
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.bits.partial_cmp(&rhs.bits)
    }
}

macro_rules! q {
    ($bits:ident, $dbits:ident, $limit:ident) => {
        impl<FRAC> Q<$bits, FRAC>
        where
            FRAC: Cmp<U0, Output = Greater> +
                Cmp<$limit, Output = Less> +
                Unsigned,
        {
            /// Creates a fixed point number from the bit pattern `$bits`
            pub fn from_bits(bits: $bits) -> Self {
                Q { bits: bits, _marker: PhantomData }
            }
        }

        impl<FRAC> Q<$bits, FRAC> {
            /// Returns the bit pattern that represents `self`
            pub fn into_bits(self) -> $bits {
                self.bits
            }
        }

        impl<FRAC> Q<$bits, FRAC>
        where
            FRAC: Unsigned
        {
            fn fbits() -> u8 {
                FRAC::to_u8()
            }
        }

        #[cfg(feature = "try_from")]
        impl<FRAC> TryFrom<f32> for Q<$bits, FRAC>
        where
            FRAC: Cmp<U0, Output = Greater> +
                Cmp<$limit, Output = Less> +
                Unsigned,
        {
            type Error = Error;

            fn try_from(x: f32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl<FRAC> cast::From<f32> for Q<$bits, FRAC>
        where
            FRAC: Cmp<U0, Output = Greater> +
                Cmp<$limit, Output = Less> +
                Unsigned,
        {
            type Output = Result<Self, Error>;

            fn cast(x: f32) -> Self::Output {
                $bits(x * f32(1u64 << FRAC::to_u8())).map(|bits| {
                    Q { bits: bits, _marker: PhantomData }
                })
            }
        }

        #[cfg(feature = "try_from")]
        impl<FRAC> TryFrom<f64> for Q<$bits, FRAC>
        where
            FRAC: Cmp<U0, Output = Greater> +
                Cmp<$limit, Output = Less> +
                Unsigned,
        {
            type Error = Error;

            fn try_from(x: f64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl<FRAC> cast::From<f64> for Q<$bits, FRAC>
        where
            FRAC: Cmp<U0, Output = Greater> +
                Cmp<$limit, Output = Less> +
                Unsigned,
        {
            type Output = Result<Self, Error>;

            fn cast(x: f64) -> Self::Output {
                $bits(x * f64(1u64 << FRAC::to_u8())).map(|bits| {
                    Q { bits: bits, _marker: PhantomData }
                })
            }
        }

        // Covered by core's `impl<T> From<T> for T`:

        impl<FRAC> cast::From<Q<$bits, FRAC>> for Q<$bits, FRAC> {
            type Output = Self;

            fn cast(x: Q<$bits, FRAC>) -> Self::Output {
                x
            }
        }

        impl<FRAC> From<Q<$bits, FRAC>> for f32
        where
            FRAC: Unsigned,
        {
            fn from(x: Q<$bits, FRAC>) -> Self {
                Self::cast(x)
            }
        }

        impl<FRAC> cast::From<Q<$bits, FRAC>> for f32
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn cast(x: Q<$bits, FRAC>) -> Self::Output {
                f32(x.bits) / f32(1u64 << FRAC::to_u8())
            }
        }

        impl<FRAC> From<Q<$bits, FRAC>> for f64
        where
            FRAC: Unsigned,
        {
            fn from(x: Q<$bits, FRAC>) -> Self {
                Self::cast(x)
            }
        }

        impl<FRAC> cast::From<Q<$bits, FRAC>> for f64
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn cast(x: Q<$bits, FRAC>) -> Self::Output {
                f64(x.bits) / f64(1u64 << FRAC::to_u8())
            }
        }

        impl<FRAC> fmt::Debug for Q<$bits, FRAC> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_struct("Q").field("bits", &self.bits).finish()
            }
        }

        impl<FRAC> fmt::Display for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f64(*self).fmt(f)
            }
        }

        impl<FRAC> ops::Add for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                Q { bits: self.bits + rhs.bits, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Add<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn add(self, rhs: $bits) -> Self {
                Q {
                    bits: self.bits + (rhs << FRAC::to_u8()),
                    _marker: PhantomData,
                }
            }
        }

        impl<FRAC> ops::Add<Q<$bits, FRAC>> for $bits
        where
            FRAC: Unsigned,
        {
            type Output = Q<$bits, FRAC>;

            fn add(self, rhs: Q<$bits, FRAC>) -> Self::Output {
                rhs + self
            }
        }

        impl<FRAC> ops::AddAssign for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<FRAC> ops::AddAssign<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn add_assign(&mut self, rhs: $bits) {
                *self = *self + rhs;
            }
        }

        impl<FRAC> ops::Div for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn div(self, rhs: Self) -> Self {
                let d = ($dbits(self.bits) << FRAC::to_u8()) / $dbits(rhs.bits);

                let bits = match () {
                    #[cfg(debug_assertions)]
                    () => $bits(d).expect("attempt to multiply with overflow"),
                    #[cfg(not(debug_assertions))]
                    () => d as $bits,
                };

                Q { bits: bits, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Div<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn div(self, rhs: $bits) -> Self {
                Q { bits: self.bits / rhs, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::DivAssign for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl<FRAC> ops::DivAssign<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn div_assign(&mut self, rhs: $bits) {
                *self = *self / rhs;
            }
        }

        impl<FRAC> ops::Mul for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                let p = ($dbits(self.bits) * $dbits(rhs.bits)) >> FRAC::to_u8();

                let bits = match () {
                    #[cfg(debug_assertions)]
                    () => $bits(p).expect("attempt to multiply with overflow"),
                    #[cfg(not(debug_assertions))]
                    () => p as $bits,
                };

                Q { bits: bits, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Mul<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn mul(self, rhs: $bits) -> Self {
                Q { bits: self.bits * rhs, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Mul<Q<$bits, FRAC>> for $bits
        where
            FRAC: Unsigned,
        {
            type Output = Q<$bits, FRAC>;

            fn mul(self, rhs: Q<$bits, FRAC>) -> Q<$bits, FRAC> {
                rhs * self
            }
        }

        impl<FRAC> ops::MulAssign for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl<FRAC> ops::MulAssign<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn mul_assign(&mut self, rhs: $bits) {
                *self = *self * rhs;
            }
        }

        impl<FRAC> ops::Neg for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn neg(self) -> Self {
                Q { bits: -self.bits, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Rem<Q<$bits, FRAC>> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn rem(self, rhs: Self) -> Self {
                let d = ($dbits(self.bits) << FRAC::to_u8()) % $dbits(rhs.bits);

                let bits = match () {
                    #[cfg(debug_assertions)]
                    () => $bits(d).expect("attempt to multiply with overflow"),
                    #[cfg(not(debug_assertions))]
                    () => d as $bits,
                };

                Q { bits: bits, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Rem<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn rem(self, rhs: $bits) -> Self {
                Q { bits: self.bits % rhs, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Shl<u8> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn shl(self, rhs: u8) -> Self {
                Q { bits: self.bits << rhs, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Shr<u8> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn shr(self, rhs: u8) -> Self {
                Q { bits: self.bits >> rhs, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Sub for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                Q { bits: self.bits - rhs.bits, _marker: PhantomData }
            }
        }

        impl<FRAC> ops::Sub<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            type Output = Self;

            fn sub(self, rhs: $bits) -> Self {
                Q {
                    bits: self.bits - (rhs << FRAC::to_u8()),
                    _marker: PhantomData,
                }
            }
        }

        impl<FRAC> ops::Sub<Q<$bits, FRAC>> for $bits
        where
            FRAC: Unsigned,
        {
            type Output = Q<$bits, FRAC>;

            fn sub(self, rhs: Q<$bits, FRAC>) -> Q<$bits, FRAC> {
                -(rhs - self)
            }
        }

        impl<FRAC> ops::SubAssign for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<FRAC> ops::SubAssign<$bits> for Q<$bits, FRAC>
        where
            FRAC: Unsigned,
        {
            fn sub_assign(&mut self, rhs: $bits) {
                *self = *self - rhs;
            }
        }
    }
}

q!(i8, i16, U8);
q!(i16, i32, U16);
q!(i32, i64, U32);

// Type alias + const new + checked cast function
macro_rules! ty {
    ($ty:ident, $bits:ident, $frac:ident, $ifrac:expr) => {
        /// Fixed point number
        pub type $ty = Q<$bits, $frac>;

        impl $ty {
            /// Converts a `f64` into a fixed point number
            #[cfg(feature = "const-fn")]
            pub const fn new(x: f64) -> Self {
                Q {
                    bits: (x * (1 << $ifrac) as f64) as $bits,
                    _marker: PhantomData,
                }
            }
        }

        /// Checked cast
        #[allow(non_snake_case)]
        pub fn $ty<T>(x: T) -> <$ty as cast::From<T>>::Output
        where
            $ty: cast::From<T>,
        {
            <$ty as cast::From<T>>::cast(x)
        }
    };
    ($ty:ident, $bits:ident, $frac:ident, $ifrac:expr, cordic) => {
        ty!($ty, $bits, $frac, $ifrac);

        impl $ty {
            /// Turns the cartesian coordinates `(x, y)` (where `x = self`) into
            /// polar coordinates.
            ///
            /// The returned value is `((x * x + y * y).sqrt(), atan2(y, x))`
            pub fn polar(self, y: Self) -> (Self, I3F29) {
                #![allow(non_snake_case)]

                #[cfg(feature = "const-fn")]
                const K: $ty = $ty::new(0.3687561270769006);

                #[cfg(not(feature = "const-fn"))]
                let K = $ty(0.3687561270769006_f64).unwrap();
                let (r, theta) = cordic(self.into_bits(),
                                        y.into_bits());

                (K * $ty::from_bits(r), theta)
            }

            /// computes the arctan of self
            pub fn atan(self) -> I3F29 {
                cordic(I16F16::one().into_bits(), self.into_bits()).1
            }

            /// sine function in radians
            pub fn sin(self) -> I8F24 {
                let two_pi  = $ty(6.283185307179586476925_f64).unwrap();
                let pi      = $ty(3.141592653589793238462_f64).unwrap();
                let pi_half = $ty(1.570796326794896619231_f64).unwrap();

                let mut angle = self;

                //wraparound
                while angle > pi {
                    angle -= two_pi;
                }
                while angle < -pi {
                    angle += two_pi;
                }
                //mirror
                if angle > pi_half {
                    angle = pi_half - (angle - pi_half);
                }
                if angle < -pi_half {
                    angle = -pi_half - (angle + pi_half);
                }

                //FIXME: find correction factor for constant iterations
                // x0= 1/K with K ~ 1.647 for infinite iterations
                let x = I8F24(0.607253_f64).unwrap();

                // FIXME: typecast doesn't work. only works for I8F24
                let (_x,y) = cordic_rotation(x, I8F24::zero(), I8F24::from_bits(angle.into_bits()));
                y 
            }

            /// cosine function in radians
            pub fn cos(self) -> I8F24 {
                let pi_half = $ty(1.570796326794896619231_f64).unwrap();
                let angle = self + pi_half;
                angle.sin()
            }

            /// tangent function in radians
            pub fn tan(self) -> I8F24 {
                let angle = self << 1;
                angle.sin() / ( 1 + angle.cos())
            }

            /// square root of self
            pub fn sqrt(self) -> Result<Self, ()> {
                let mut invert = false;
                let mut operand = self;
                if self < Self::zero() {
                    return Err(());
                };
                if operand == Self::zero() || operand == Self::one() {
                    return Ok(self);
                };
                if operand < Self::one() && operand.into_bits() > 6 {
                    invert = true;
                    operand = Self::one() / operand;
                }
                // Newton iterations
                let mut l = (operand >> 1) + 1;
                for _i in 0..$ifrac {
                    l = (l + operand / l) >> 1;
                }
                if invert {
                    l = Self::one() / l; 
                }
                Ok(l)
            }

            /// exponential function e^(self)
            pub fn exp(self) -> Self {
                #![allow(non_snake_case)]

                #[cfg(not(feature = "const-fn"))]
                let E : Self = $ty(2.718281828459045235360287471_f64).unwrap();
                #[cfg(feature = "const-fn")]
                const E : Self = $ty(2.718281828459045235360287471_f64).unwrap();

                let mut operand = self;

                if operand == Self::zero() {
                    return Self::one();
                };
                if operand == Self::one() {
                    return E;
                };
                let neg = operand < Self::zero();
                if neg {
                    operand = -operand; 
                };

                let mut result = operand + Self::one();
                let mut term = operand;

                for i in (2..30) {
                    term = term * operand / $ty(i as u32).unwrap();
                    result += term;
                    //if term < 500 && (i > 15 || term < $ty(20i32).unwrap()) {
                    //    break;
                    //};

                }
                if neg {
                    result = Self::one() / result;
                }
                result
            }

            /// power
            pub fn pow(self, exp: Self) -> Self {
                if exp == Self::zero() {
                    return self;
                };
                if exp < Self::zero() {
                    return Self::zero();
                }
                (self.ln().unwrap() * exp).exp()
            }

            /// right-shift with rounding
            fn rs(self) -> Self {
                let x = self.into_bits();
                Self::from_bits((x >> 1) + (x & 1))
            }

            /// base 2 logarithm assuming self >=1
            fn log2_inner(self) -> Self {
                #![allow(non_snake_case)]

                #[cfg(not(feature = "const-fn"))]
                let TWO: $ty = $ty(2.0_f64).unwrap();
                #[cfg(feature = "const-fn")]
                const TWO: $ty = $ty(2.0_f64).unwrap();

                let mut x = self;
                let mut result = 0i32;  

                while x >= TWO {
                    result += 1;
                    x = x.rs();
                }
                
                if x == Self::one() {
                    return Self::from_bits(result << 16);
                };

                for _i in (0..16).rev() {
                    x *= x;
                    result = result << 1;
                    if x >= TWO {
                        result = result | 1;
                        x = x.rs();
                    }
                } 
                Self::from_bits(result)
            }

            /// base 2 logarithm 
            pub fn log2(self) -> Result<Self, ()> {
                if self <= Self::zero() {
                    return Err(())
                };
                if self < Self::one() {
                    let inverse = Self::one() / self;
                    return Ok(-inverse.log2_inner());
                };
                return Ok(self.log2_inner());
            }

            /// natural logarithm
            pub fn ln(self) -> Result<Self, ()> {    
                Ok(self.log2()? * $ty(0.69314718055994530941723212_f64).unwrap())
            }    

        }
    };
}

ty!(I7F1, i8, U1, 1);
ty!(I6F2, i8, U2, 2);
ty!(I5F3, i8, U3, 3);
ty!(I4F4, i8, U4, 4);
ty!(I3F5, i8, U5, 5);
ty!(I2F6, i8, U6, 6);
ty!(I1F7, i8, U7, 7);

ty!(I15F1, i16, U1, 1);
ty!(I14F2, i16, U2, 2);
ty!(I13F3, i16, U3, 3);
ty!(I12F4, i16, U4, 4);
ty!(I11F5, i16, U5, 5);
ty!(I10F6, i16, U6, 6);
ty!(I9F7, i16, U7, 7);
ty!(I8F8, i16, U8, 8);
ty!(I7F9, i16, U9, 9);
ty!(I6F10, i16, U10, 10);
ty!(I5F11, i16, U11, 11);
ty!(I4F12, i16, U12, 12);
ty!(I3F13, i16, U13, 13);
ty!(I2F14, i16, U14, 14);
ty!(I1F15, i16, U15, 15);

ty!(I31F1, i32, U1, 1);
ty!(I30F2, i32, U2, 2);
ty!(I29F3, i32, U3, 3);
ty!(I28F4, i32, U4, 4);
ty!(I27F5, i32, U5, 5);
ty!(I26F6, i32, U6, 6);
ty!(I25F7, i32, U7, 7);
ty!(I24F8, i32, U8, 8);
ty!(I23F9, i32, U9, 9);
ty!(I22F10, i32, U10, 10);
ty!(I21F11, i32, U11, 11);
ty!(I20F12, i32, U12, 12);
ty!(I19F13, i32, U13, 13);
ty!(I18F14, i32, U14, 14);
ty!(I17F15, i32, U15, 15);
ty!(I16F16, i32, U16, 16, cordic);
ty!(I15F17, i32, U17, 17, cordic);
ty!(I14F18, i32, U18, 18, cordic);
ty!(I13F19, i32, U19, 19, cordic);
ty!(I12F20, i32, U20, 20, cordic);
ty!(I11F21, i32, U21, 21, cordic);
ty!(I10F22, i32, U22, 22, cordic);
ty!(I9F23, i32, U23, 23, cordic);
ty!(I8F24, i32, U24, 24, cordic);
ty!(I7F25, i32, U25, 25, cordic);
ty!(I6F26, i32, U26, 26, cordic);
ty!(I5F27, i32, U27, 27, cordic);
ty!(I4F28, i32, U28, 28, cordic);
ty!(I3F29, i32, U29, 29, cordic);
ty!(I2F30, i32, U30, 30, cordic);
ty!(I1F31, i32, U31, 31, cordic);

/// CORDIC in vectoring mode
fn cordic(mut x: i32, mut y: i32) -> (i32, I3F29) {
    #![allow(non_snake_case)]

    // NOTE `(2 ^ -i).atan()` for `i = 0, 0, 1, 1, 2, 2, ..`
    #[cfg(feature = "const-fn")]
    const ANGLES: [I3F29; 62] = [
        I3F29::new(0.7853981633974483),
        I3F29::new(0.7853981633974483),
        I3F29::new(0.4636476090008061),
        I3F29::new(0.4636476090008061),
        I3F29::new(0.24497866312686414),
        I3F29::new(0.24497866312686414),
        I3F29::new(0.12435499454676144),
        I3F29::new(0.12435499454676144),
        I3F29::new(0.06241880999595735),
        I3F29::new(0.06241880999595735),
        I3F29::new(0.031239833430268277),
        I3F29::new(0.031239833430268277),
        I3F29::new(0.015623728620476831),
        I3F29::new(0.015623728620476831),
        I3F29::new(0.007812341060101111),
        I3F29::new(0.007812341060101111),
        I3F29::new(0.0039062301319669718),
        I3F29::new(0.0039062301319669718),
        I3F29::new(0.0019531225164788188),
        I3F29::new(0.0019531225164788188),
        I3F29::new(0.0009765621895593195),
        I3F29::new(0.0009765621895593195),
        I3F29::new(0.0004882812111948983),
        I3F29::new(0.0004882812111948983),
        I3F29::new(0.00024414062014936177),
        I3F29::new(0.00024414062014936177),
        I3F29::new(0.00012207031189367021),
        I3F29::new(0.00012207031189367021),
        I3F29::new(0.00006103515617420877),
        I3F29::new(0.00006103515617420877),
        I3F29::new(0.000030517578115526096),
        I3F29::new(0.000030517578115526096),
        I3F29::new(0.000015258789061315762),
        I3F29::new(0.000015258789061315762),
        I3F29::new(0.00000762939453110197),
        I3F29::new(0.00000762939453110197),
        I3F29::new(0.000003814697265606496),
        I3F29::new(0.000003814697265606496),
        I3F29::new(0.000001907348632810187),
        I3F29::new(0.000001907348632810187),
        I3F29::new(0.0000009536743164059608),
        I3F29::new(0.0000009536743164059608),
        I3F29::new(0.00000047683715820308884),
        I3F29::new(0.00000047683715820308884),
        I3F29::new(0.00000023841857910155797),
        I3F29::new(0.00000023841857910155797),
        I3F29::new(0.00000011920928955078068),
        I3F29::new(0.00000011920928955078068),
        I3F29::new(0.00000005960464477539055),
        I3F29::new(0.00000005960464477539055),
        I3F29::new(0.000000029802322387695303),
        I3F29::new(0.000000029802322387695303),
        I3F29::new(0.000000014901161193847655),
        I3F29::new(0.000000014901161193847655),
        I3F29::new(0.000000007450580596923828),
        I3F29::new(0.000000007450580596923828),
        I3F29::new(0.000000003725290298461914),
        I3F29::new(0.000000003725290298461914),
        I3F29::new(0.000000001862645149230957),
        I3F29::new(0.000000001862645149230957),
        I3F29::new(0.0000000009313225746154785),
        I3F29::new(0.0000000009313225746154785),
    ];

    #[cfg(feature = "const-fn")]
    const ZERO: I3F29 = I3F29::new(0.);

    #[cfg(not(feature = "const-fn"))]
    let ANGLES = [
        I3F29(0.7853981633974483_f64).unwrap(),
        I3F29(0.7853981633974483_f64).unwrap(),
        I3F29(0.4636476090008061_f64).unwrap(),
        I3F29(0.4636476090008061_f64).unwrap(),
        I3F29(0.24497866312686414_f64).unwrap(),
        I3F29(0.24497866312686414_f64).unwrap(),
        I3F29(0.12435499454676144_f64).unwrap(),
        I3F29(0.12435499454676144_f64).unwrap(),
        I3F29(0.06241880999595735_f64).unwrap(),
        I3F29(0.06241880999595735_f64).unwrap(),
        I3F29(0.031239833430268277_f64).unwrap(),
        I3F29(0.031239833430268277_f64).unwrap(),
        I3F29(0.015623728620476831_f64).unwrap(),
        I3F29(0.015623728620476831_f64).unwrap(),
        I3F29(0.007812341060101111_f64).unwrap(),
        I3F29(0.007812341060101111_f64).unwrap(),
        I3F29(0.0039062301319669718_f64).unwrap(),
        I3F29(0.0039062301319669718_f64).unwrap(),
        I3F29(0.0019531225164788188_f64).unwrap(),
        I3F29(0.0019531225164788188_f64).unwrap(),
        I3F29(0.0009765621895593195_f64).unwrap(),
        I3F29(0.0009765621895593195_f64).unwrap(),
        I3F29(0.0004882812111948983_f64).unwrap(),
        I3F29(0.0004882812111948983_f64).unwrap(),
        I3F29(0.00024414062014936177_f64).unwrap(),
        I3F29(0.00024414062014936177_f64).unwrap(),
        I3F29(0.00012207031189367021_f64).unwrap(),
        I3F29(0.00012207031189367021_f64).unwrap(),
        I3F29(0.00006103515617420877_f64).unwrap(),
        I3F29(0.00006103515617420877_f64).unwrap(),
        I3F29(0.000030517578115526096_f64).unwrap(),
        I3F29(0.000030517578115526096_f64).unwrap(),
        I3F29(0.000015258789061315762_f64).unwrap(),
        I3F29(0.000015258789061315762_f64).unwrap(),
        I3F29(0.00000762939453110197_f64).unwrap(),
        I3F29(0.00000762939453110197_f64).unwrap(),
        I3F29(0.000003814697265606496_f64).unwrap(),
        I3F29(0.000003814697265606496_f64).unwrap(),
        I3F29(0.000001907348632810187_f64).unwrap(),
        I3F29(0.000001907348632810187_f64).unwrap(),
        I3F29(0.0000009536743164059608_f64).unwrap(),
        I3F29(0.0000009536743164059608_f64).unwrap(),
        I3F29(0.00000047683715820308884_f64).unwrap(),
        I3F29(0.00000047683715820308884_f64).unwrap(),
        I3F29(0.00000023841857910155797_f64).unwrap(),
        I3F29(0.00000023841857910155797_f64).unwrap(),
        I3F29(0.00000011920928955078068_f64).unwrap(),
        I3F29(0.00000011920928955078068_f64).unwrap(),
        I3F29(0.00000005960464477539055_f64).unwrap(),
        I3F29(0.00000005960464477539055_f64).unwrap(),
        I3F29(0.000000029802322387695303_f64).unwrap(),
        I3F29(0.000000029802322387695303_f64).unwrap(),
        I3F29(0.000000014901161193847655_f64).unwrap(),
        I3F29(0.000000014901161193847655_f64).unwrap(),
        I3F29(0.000000007450580596923828_f64).unwrap(),
        I3F29(0.000000007450580596923828_f64).unwrap(),
        I3F29(0.000000003725290298461914_f64).unwrap(),
        I3F29(0.000000003725290298461914_f64).unwrap(),
        I3F29(0.000000001862645149230957_f64).unwrap(),
        I3F29(0.000000001862645149230957_f64).unwrap(),
        I3F29(0.0000000009313225746154785_f64).unwrap(),
        I3F29(0.0000000009313225746154785_f64).unwrap(),
    ];

    #[cfg(not(feature = "const-fn"))]
    let ZERO = I3F29(0_f64).unwrap();

    let mut z = ZERO;

    for (angle, i) in ANGLES.iter().cloned().zip(0..) {
        let i = i / 2;
        let prev_x = x;
        if y == 0 {
            break;
        } else if y > 0 {
            x += y >> i;
            y -= prev_x >> i;
            z += angle;
        } else {
            x -= y >> i;
            y += prev_x >> i;
            z -= angle;
        }
    }

    (x, z)
}

//FIXME: write macro to avoid redundant code
/// CORDIC in rotation mode. 
fn cordic_rotation(mut x: I8F24, mut y: I8F24, mut z: I8F24) -> (I8F24, I8F24) {
    #![allow(non_snake_case)]

    // NOTE `(2 ^ -i).atan()` for `i = 0, 1, 2, ..`
    #[cfg(feature = "const-fn")]
    const ANGLES: [I8F24; 31] = [
        I8F24::new(0.7853981633974483),
        I8F24::new(0.4636476090008061),
        I8F24::new(0.24497866312686414),
        I8F24::new(0.12435499454676144),
        I8F24::new(0.06241880999595735),
        I8F24::new(0.031239833430268277),
        I8F24::new(0.015623728620476831),
        I8F24::new(0.007812341060101111),
        I8F24::new(0.0039062301319669718),
        I8F24::new(0.0019531225164788188),
        I8F24::new(0.0009765621895593195),
        I8F24::new(0.0004882812111948983),
        I8F24::new(0.00024414062014936177),
        I8F24::new(0.00012207031189367021),
        I8F24::new(0.00006103515617420877),
        I8F24::new(0.000030517578115526096),
        I8F24::new(0.000015258789061315762),
        I8F24::new(0.00000762939453110197),
        I8F24::new(0.000003814697265606496),
        I8F24::new(0.000001907348632810187),
        I8F24::new(0.0000009536743164059608),
        I8F24::new(0.00000047683715820308884),
        I8F24::new(0.00000023841857910155797),
        I8F24::new(0.00000011920928955078068),
        I8F24::new(0.00000005960464477539055),
        I8F24::new(0.000000029802322387695303),
        I8F24::new(0.000000014901161193847655),
        I8F24::new(0.000000007450580596923828),
        I8F24::new(0.000000003725290298461914),
        I8F24::new(0.000000001862645149230957),
        I8F24::new(0.0000000009313225746154785),
    ];

    #[cfg(feature = "const-fn")]
    const ZERO: I8F24 = I8F24::new(0.);

    #[cfg(not(feature = "const-fn"))]
    let ANGLES = [
        I8F24(0.7853981633974483_f64).unwrap(),
        I8F24(0.4636476090008061_f64).unwrap(),
        I8F24(0.24497866312686414_f64).unwrap(),
        I8F24(0.12435499454676144_f64).unwrap(),
        I8F24(0.06241880999595735_f64).unwrap(),
        I8F24(0.031239833430268277_f64).unwrap(),
        I8F24(0.015623728620476831_f64).unwrap(),
        I8F24(0.007812341060101111_f64).unwrap(),
        I8F24(0.0039062301319669718_f64).unwrap(),
        I8F24(0.0019531225164788188_f64).unwrap(),
        I8F24(0.0009765621895593195_f64).unwrap(),
        I8F24(0.0004882812111948983_f64).unwrap(),
        I8F24(0.00024414062014936177_f64).unwrap(),
        I8F24(0.00012207031189367021_f64).unwrap(),
        I8F24(0.00006103515617420877_f64).unwrap(),
        I8F24(0.000030517578115526096_f64).unwrap(),
        I8F24(0.000015258789061315762_f64).unwrap(),
        I8F24(0.00000762939453110197_f64).unwrap(),
        I8F24(0.000003814697265606496_f64).unwrap(),
        I8F24(0.000001907348632810187_f64).unwrap(),
        I8F24(0.0000009536743164059608_f64).unwrap(),
        I8F24(0.00000047683715820308884_f64).unwrap(),
        I8F24(0.00000023841857910155797_f64).unwrap(),
        I8F24(0.00000011920928955078068_f64).unwrap(),
        I8F24(0.00000005960464477539055_f64).unwrap(),
        I8F24(0.000000029802322387695303_f64).unwrap(),
        I8F24(0.000000014901161193847655_f64).unwrap(),
        I8F24(0.000000007450580596923828_f64).unwrap(),
        I8F24(0.000000003725290298461914_f64).unwrap(),
        I8F24(0.000000001862645149230957_f64).unwrap(),
        I8F24(0.0000000009313225746154785_f64).unwrap(),
    ];

    #[cfg(not(feature = "const-fn"))]
    let ZERO = I8F24(0_f64).unwrap();
    
    for (angle, i) in ANGLES.iter().cloned().zip(0..) {
        //if z == ZERO {
        //    break;
        //};
        if i >= 24 {
            break;
        }
        let prev_x = x;
        if z < ZERO {
            x += y >> i;
            y -= prev_x >> i;
            z += angle;
        } else {
            x -= y >> i;
            y += prev_x >> i;
            z -= angle;
        }
        //k *= (I8F24::one() + (I8F24(2i32).unwrap() >> i)).sqrt().unwrap()
    }

    (x, y)
}



macro_rules! cast {
    (+ $largest32:ident,
     $(+ $large32:ident,)*
     $(| $medium32:ident $medium16:ident,)+
     $(- $small32:ident $small16:ident $small8:ident,)+) => {
         #[cfg(feature = "try_from")]
         impl TryFrom<$largest32> for isize {
             type Error = Infallible;

             fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                 Self::cast(x)
             }
         }

        impl cast::From<$largest32> for isize {
            type Output = Self;

            fn cast(x: $largest32) -> Self::Output {
                isize(i32(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<isize> for $largest32 {
            type Error = TryFromIntError;

            fn try_from(x: isize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<isize> for $largest32 {
            type Output = Result<$largest32, Error>;

            fn cast(x: isize) -> Self::Output {
                i16(x).map($largest32)
            }
        }

        impl From<$largest32> for i64 {
            fn from(x: $largest32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for i64 {
            type Output = Self;

            fn cast(x: $largest32) -> Self::Output {
                i64(i32(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i64> for $largest32 {
            type Error = TryFromIntError;

            fn try_from(x: i64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i64> for $largest32 {
            type Output = Result<$largest32, Error>;

            fn cast(x: i64) -> Self::Output {
                i16(x).map($largest32)
            }
        }

        impl From<$largest32> for i32 {
            fn from(x: $largest32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for i32 {
            type Output = Self;

            fn cast(x: $largest32) -> i32 {
                x.bits >> $largest32::fbits()
            }

            }
        #[cfg(feature = "try_from")]
        impl TryFrom<i32> for $largest32 {
            type Error = TryFromIntError;

            fn try_from(x: i32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i32> for $largest32 {
            type Output = Result<$largest32, Error>;

            fn cast(x: i32) -> Self::Output {
                i16(x).map($largest32)
            }
        }

        // Special case handled below
        // impl cast::From<$largest32> for i16 {}

        impl From<i16> for $largest32 {
            fn from(x: i16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<i16> for $largest32 {
            type Output = $largest32;

            fn cast(x: i16) -> Self::Output {
                $largest32::from_bits(i32(x) << $largest32::fbits())
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$largest32> for i8 {
            type Error = TryFromIntError;

            fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for i8 {
            type Output = Result<Self, Error>;

            fn cast(x: $largest32) -> Self::Output {
                i8(i32(x))
            }
        }

        impl From<i8> for $largest32 {
            fn from(x: i8) -> Self {
                Self::cast(x)
        }
        }

        impl cast::From<i8> for $largest32 {
            type Output = $largest32;

            fn cast(x: i8) -> Self::Output {
                $largest32(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$largest32> for usize {
            type Error = TryFromIntError;

            fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for usize {
            type Output = Result<Self, Error>;

            fn cast(x: $largest32) -> Self::Output {
                usize(i32(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<usize> for $largest32 {
            type Error = TryFromIntError;

            fn try_from(x: usize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<usize> for $largest32 {
            type Output = Result<$largest32, Error>;

            fn cast(x: usize) -> Self::Output {
                i16(x).map($largest32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$largest32> for u64 {
            type Error = TryFromIntError;

            fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for u64 {
            type Output = Result<Self, Error>;

            fn cast(x: $largest32) -> Self::Output {
                u64(i32(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u64> for $largest32 {
            type Error = TryFromIntError;

            fn try_from(x: u64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u64> for $largest32 {
            type Output = Result<$largest32, Error>;

            fn cast(x: u64) -> Self::Output {
                i16(x).map($largest32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$largest32> for u32 {
            type Error = TryFromIntError;

            fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for u32 {
            type Output = Result<Self, Error>;

            fn cast(x: $largest32) -> Self::Output {
                u32(i32(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u32> for $largest32 {
            type Error = TryFromIntError;

            fn try_from(x: u32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u32> for $largest32 {
            type Output = Result<$largest32, Error>;

            fn cast(x: u32) -> Self::Output {
                i16(x).map($largest32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$largest32> for u16 {
            type Error = TryFromIntError;

            fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for u16 {
            type Output = Result<Self, Error>;

            fn cast(x: $largest32) -> Self::Output {
                u16(i32(x))
            }
        }

        // Special case handled below
        // impl cast::From<u16> for $largest32 {}

        #[cfg(feature = "try_from")]
        impl TryFrom<$largest32> for u8 {
            type Error = TryFromIntError;

            fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$largest32> for u8 {
            type Output = Result<Self, Error>;

            fn cast(x: $largest32) -> Self::Output {
                u8(i32(x))
            }
        }

        impl From<u8> for $largest32 {
            fn from(x: u8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<u8> for $largest32 {
            type Output = $largest32;

            fn cast(x: u8) -> Self::Output {
                $largest32(i16(x))
            }
        }

        // Casts between fixed point numbers
        $(
            impl From<$large32> for $largest32 {
                fn from(x: $large32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$large32> for $largest32 {
                type Output = $largest32;

                fn cast(x: $large32) -> Self::Output {
                    $largest32::from_bits(x.bits >>
                                          ($large32::fbits() -
                                           $largest32::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$largest32> for $large32 {
                type Error = TryFromIntError;

                fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$largest32> for $large32 {
                type Output = Result<$large32, Error>;

                fn cast(x: $largest32) -> Self::Output {
                    i32(i64(x.bits) <<
                        ($large32::fbits() -
                         $largest32::fbits()))
                        .map($large32::from_bits)
                }
            }
        )*

        $(
            impl From<$medium32> for $largest32 {
                fn from(x: $medium32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium32> for $largest32 {
                type Output = $largest32;

                fn cast(x: $medium32) -> Self::Output {
                    $largest32::from_bits(x.bits >>
                                          ($medium32::fbits() -
                                           $largest32::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$largest32> for $medium32 {
                type Error = TryFromIntError;

                fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$largest32> for $medium32 {
                type Output = Result<$medium32, Error>;

                fn cast(x: $largest32) -> Self::Output {
                    i32(i64(x.bits) <<
                        ($medium32::fbits() -
                         $largest32::fbits()))
                        .map($medium32::from_bits)
                }
            }

            impl From<$medium16> for $largest32 {
                fn from(x: $medium16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium16> for $largest32 {
                type Output = $largest32;

                fn cast(x: $medium16) -> Self::Output {
                    $largest32($medium32(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$largest32> for $medium16 {
                type Error = TryFromIntError;

                fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$largest32> for $medium16 {
                type Output = Result<$medium16, Error>;

                fn cast(x: $largest32) -> Self::Output {
                    $medium32(x).map($medium16)
                }
            }
        )*

        $(
            impl From<$small32> for $largest32 {
                fn from(x: $small32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small32> for $largest32 {
                type Output = $largest32;

                fn cast(x: $small32) -> Self::Output {
                    $largest32::from_bits(x.bits >>
                                          ($small32::fbits() -
                                           $largest32::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$largest32> for $small32 {
                type Error = TryFromIntError;

                fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$largest32> for $small32 {
                type Output = Result<$small32, Error>;

                fn cast(x: $largest32) -> Self::Output {
                    i32(i64(x.bits) <<
                        ($small32::fbits() -
                         $largest32::fbits()))
                        .map($small32::from_bits)
                }

                }
            impl From<$small16> for $largest32 {
                fn from(x: $small16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small16> for $largest32 {
                type Output = $largest32;

                fn cast(x: $small16) -> Self::Output {
                    $largest32($small32(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$largest32> for $small16 {
                type Error = TryFromIntError;

                fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$largest32> for $small16 {
                type Output = Result<$small16, Error>;

                fn cast(x: $largest32) -> Self::Output {
                    $small32(x).map($small16)
                }
            }

            impl From<$small8> for $largest32 {
                fn from(x: $small8) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small8> for $largest32 {
                type Output = $largest32;

                fn cast(x: $small8) -> Self::Output {
                    $largest32($small32(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$largest32> for $small8 {
                type Error = TryFromIntError;

                fn try_from(x: $largest32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$largest32> for $small8 {
                type Output = Result<$small8, Error>;

                fn cast(x: $largest32) -> Self::Output {
                    $small32(x).map($small8)
                }
            }
        )*

        cast! {
            $(+ $large32,)*
            $(| $medium32 $medium16,)+
            $(- $small32 $small16 $small8,)+
        }
    };

    (| $large32:ident $large16:ident,
     $(| $medium32:ident $medium16:ident,)*
     $(- $small32:ident $small16:ident $small8:ident,)+) => {
         #[cfg(feature = "try_from")]
         impl TryFrom<$large32> for isize {
             type Error = Infallible;

             fn try_from(x: $large32) -> Result<Self, Self::Error> {
                 Self::cast(x)
             }
         }

        impl cast::From<$large32> for isize {
            type Output = Self;

            fn cast(x: $large32) -> Self::Output {
                isize(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<isize> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: isize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<isize> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: isize) -> Self::Output {
                i8(x).map($large32)
            }
        }

        impl From<$large32> for i64 {
            fn from(x: $large32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for i64 {
            type Output = Self;

            fn cast(x: $large32) -> Self::Output {
                i64(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i64> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: i64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i64> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: i64) -> Self::Output {
                i8(x).map($large32)
            }
        }

        impl From<$large32> for i32 {
            fn from(x: $large32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for i32 {
            type Output = Self;

            fn cast(x: $large32) -> Self::Output {
                i32(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i32> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: i32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i32> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: i32) -> Self::Output {
                i8(x).map($large32)
            }
        }

        impl From<$large32> for i16 {
            fn from(x: $large32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for i16 {
            type Output = Self;

            fn cast(x: $large32) -> Self::Output {
                (x.bits >> $large32::fbits()) as i16
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i16> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: i16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i16> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: i16) -> Self::Output {
                i8(x).map($large32)
            }
        }

        // Special case handled below
        // impl cast::From<$large32> for i8 {}

        impl From<i8> for $large32 {
            fn from(x: i8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<i8> for $large32 {
            type Output = $large32;

            fn cast(x: i8) -> Self::Output {
                $large32::from_bits(i32(x) << $large32::fbits())
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large32> for usize {
            type Error = TryFromIntError;

            fn try_from(x: $large32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for usize {
            type Output = Result<Self, Error>;

            fn cast(x: $large32) -> Self::Output {
                usize(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<usize> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: usize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<usize> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: usize) -> Self::Output {
                i8(x).map($large32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large32> for u64 {
            type Error = TryFromIntError;

            fn try_from(x: $large32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for u64 {
            type Output = Result<Self, Error>;

            fn cast(x: $large32) -> Self::Output {
                u64(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u64> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: u64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u64> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: u64) -> Self::Output {
                i8(x).map($large32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large32> for u32 {
            type Error = TryFromIntError;

            fn try_from(x: $large32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for u32 {
            type Output = Result<Self, Error>;

            fn cast(x: $large32) -> Self::Output {
                u32(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u32> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: u32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u32> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: u32) -> Self::Output {
                i8(x).map($large32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large32> for u16 {
            type Error = TryFromIntError;

            fn try_from(x: $large32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for u16 {
            type Output = Result<Self, Error>;

            fn cast(x: $large32) -> Self::Output {
                u16(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u16> for $large32 {
            type Error = TryFromIntError;

            fn try_from(x: u16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u16> for $large32 {
            type Output = Result<$large32, Error>;

            fn cast(x: u16) -> Self::Output {
                i8(x).map($large32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large32> for u8 {
            type Error = TryFromIntError;

            fn try_from(x: $large32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for u8 {
            type Output = Result<Self, Error>;

            fn cast(x: $large32) -> Self::Output {
                u8(i16(x))
            }
        }

        // Special case handled below
        // impl cast::From<u8> for $large32 {}

        #[cfg(feature = "try_from")]
        impl TryFrom<$large16> for isize {
            type Error = <isize as TryFrom<i16>>::Error;

            fn try_from(x: $large16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for isize {
            type Output = Self;

            fn cast(x: $large16) -> Self::Output {
                isize(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<isize> for $large16 {
            type Error = <i8 as TryFrom<isize>>::Error;

            fn try_from(x: isize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<isize> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: isize) -> Self::Output {
                i8(x).map($large16)
            }
        }

        impl From<$large16> for i64 {
            fn from(x: $large16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for i64 {
            type Output = Self;

            fn cast(x: $large16) -> Self::Output {
                i64(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i64> for $large16 {
            type Error = TryFromIntError;

            fn try_from(x: i64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i64> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: i64) -> Self::Output {
                i8(x).map($large16)
            }
        }

        impl From<$large16> for i32 {
            fn from(x: $large16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for i32 {
            type Output = Self;

            fn cast(x: $large16) -> Self::Output {
                i32(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i32> for $large16 {
            type Error = TryFromIntError;

            fn try_from(x: i32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i32> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: i32) -> Self::Output {
                i8(x).map($large16)
            }
        }

        impl From<$large16> for i16 {
            fn from(x: $large16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for i16 {
            type Output = Self;

            fn cast(x: $large16) -> Self::Output {
                (x.bits >> $large16::fbits()) as i16

            }
        }
        #[cfg(feature = "try_from")]
        impl TryFrom<i16> for $large16 {
            type Error = TryFromIntError;

            fn try_from(x: i16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i16> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: i16) -> Self::Output {
                i8(x).map($large16)
            }
        }

        // Special case handled below
        // impl cast::From<$large16> for i8 {}

        impl From<i8> for $large16 {
            fn from(x: i8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<i8> for $large16 {
            type Output = $large16;

            fn cast(x: i8) -> Self::Output {
                $large16::from_bits(i16(x) << $large16::fbits())
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large16> for usize {
            type Error = TryFromIntError;

            fn try_from(x: $large16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for usize {
            type Output = Result<Self, Error>;

            fn cast(x: $large16) -> Self::Output {
                usize(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<usize> for $large16 {
            type Error = TryFromIntError;

            fn try_from(x: usize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<usize> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: usize) -> Self::Output {
                i8(x).map($large16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large16> for u64 {
            type Error = TryFromIntError;

            fn try_from(x: $large16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for u64 {
            type Output = Result<Self, Error>;

            fn cast(x: $large16) -> Self::Output {
                u64(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u64> for $large16 {
            type Error = TryFromIntError;

            fn try_from(x: u64) -> Result<Self, Self::Error> {
                iSelf::cast(x)
            }
        }

        impl cast::From<u64> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: u64) -> Self::Output {
                i8(x).map($large16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large16> for u32 {
            type Error = TryFromIntError;

            fn try_from(x: $large16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for u32 {
            type Output = Result<Self, Error>;

            fn cast(x: $large16) -> Self::Output {
                u32(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u32> for $large16 {
            type Error = TryFromIntError;

            fn try_from(x: u32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u32> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: u32) -> Self::Output {
                i8(x).map($large16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large16> for u16 {
            type Error = TryFromIntError;

            fn try_from(x: $large16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for u16 {
            type Output = Result<Self, Error>;

            fn cast(x: $large16) -> Self::Output {
                u16(i16(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u16> for $large16 {
            type Error = TryFromIntError;

            fn try_from(x: u16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u16> for $large16 {
            type Output = Result<$large16, Error>;

            fn cast(x: u16) -> Self::Output {
                i8(x).map($large16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$large16> for u8 {
            type Error = TryFromIntError;

            fn try_from(x: $large16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for u8 {
            type Output = Result<Self, Error>;

            fn cast(x: $large16) -> Self::Output {
                u8(i16(x))
            }
        }

        // Special case handled below
        // impl cast::From<u8> for $large32 {}

        // Casts between fixed point numbers

        impl From<$large16> for $large32 {
            fn from(x: $large16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large16> for $large32 {
            type Output = $large32;

            fn cast(x: $large16) -> Self::Output {
                $large32::from_bits((i32(x.bits) <<
                                     ($large32::fbits() -
                                      $large16::fbits())))
            }
        }

        impl From<$large32> for $large16 {
            fn from(x: $large32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$large32> for $large16 {
            type Output = $large16;

            fn cast(x: $large32) -> Self::Output {
                $large16::from_bits((x.bits >>
                                     ($large32::fbits() -
                                      $large16::fbits())) as i16)
            }
        }

        $(
            impl From<$medium32> for $large32 {
                fn from(x: $medium32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium32> for $large32 {
                type Output = $large32;

                fn cast(x: $medium32) -> Self::Output {
                    $large32::from_bits(x.bits >>
                                        ($medium32::fbits() -
                                         $large32::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large32> for $medium32 {
                type Error = TryFromIntError;

                fn try_from(x: $large32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large32> for $medium32 {
                type Output = Result<$medium32, Error>;

                fn cast(x: $large32) -> Self::Output {
                    i32(i64(x.bits) <<
                        ($medium32::fbits() -
                         $large32::fbits()))
                        .map($medium32::from_bits)
                }
            }

            impl From<$medium16> for $large32 {
                fn from(x: $medium16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium16> for $large32 {
                type Output = $large32;

                fn cast(x: $medium16) -> Self::Output {
                    $large32($medium32(x))
                }

                }
            #[cfg(feature = "try_from")]
            impl TryFrom<$large32> for $medium16 {
                type Error = TryFromIntError;

                fn try_from(x: $large32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large32> for $medium16 {
                type Output = Result<$medium16, Error>;

                fn cast(x: $large32) -> Self::Output {
                    $medium32(x).map($medium16)
                }
            }

            impl From<$medium32> for $large16 {
                fn from(x: $medium32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium32> for $large16 {
                type Output = $large16;

                fn cast(x: $medium32) -> Self::Output {
                    $large16($medium16(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large16> for $medium32 {
                type Error = TryFromIntError;

                fn try_from(x: $large16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large16> for $medium32 {
                type Output = Result<$medium32, Error>;

                fn cast(x: $large16) -> Self::Output {
                    $medium32($large32(x))
                }
            }

            impl From<$medium16> for $large16 {
                fn from(x: $medium16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium16> for $large16 {
                type Output = $large16;

                fn cast(x: $medium16) -> Self::Output {
                    $large16::from_bits(x.bits >>
                                        ($medium16::fbits() -
                                         $large16::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large16> for $medium16 {
                type Error = TryFromIntError;

                fn try_from(x: $large16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large16> for $medium16 {
                type Output = Result<$medium16, Error>;

                fn cast(x: $large16) -> Self::Output {
                    i16(i32(x.bits) <<
                        ($medium16::fbits() - $large16::fbits()))
                        .map($medium16::from_bits)
                }
            }
        )*

        $(
            impl From<$small32> for $large32 {
                fn from(x: $small32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small32> for $large32 {
                type Output = $large32;

                fn cast(x: $small32) -> Self::Output {
                    $large32::from_bits(x.bits >>
                                        ($small32::fbits() -
                                         $large32::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large32> for $small32 {
                type Error = TryFromIntError;

                fn try_from(x: $large32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large32> for $small32 {
                type Output = Result<$small32, Error>;

                fn cast(x: $large32) -> Self::Output {
                    i32(i64(x.bits) <<
                        ($small32::fbits() -
                         $large32::fbits()))
                        .map($small32::from_bits)
                }
            }

            impl From<$small16> for $large32 {
                fn from(x: $small16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small16> for $large32 {
                type Output = $large32;

                fn cast(x: $small16) -> Self::Output {
                    $large32($small32(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large32> for $small16 {
                type Error = TryFromIntError;

                fn try_from(x: $large32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large32> for $small16 {
                type Output = Result<$small16, Error>;

                fn cast(x: $large32) -> Self::Output {
                    $small16($large16(x))
                }
            }

            impl From<$small8> for $large32 {
                fn from(x: $small8) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small8> for $large32 {
                type Output = $large32;

                fn cast(x: $small8) -> Self::Output {
                    $large32($small32(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large32> for $small8 {
                type Error = TryFromIntError;

                fn try_from(x: $large32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large32> for $small8 {
                type Output = Result<$small8, Error>;

                fn cast(x: $large32) -> Self::Output {
                    $small32(x).map($small8)
                }
            }

            impl From<$small32> for $large16 {
                fn from(x: $small32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small32> for $large16 {
                type Output = $large16;

                fn cast(x: $small32) -> Self::Output {
                    $large16($small16(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large16> for $small32 {
                type Error = TryFromIntError;

                fn try_from(x: $large16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large16> for $small32 {
                type Output = Result<$small32, Error>;

                fn cast(x: $large16) -> Self::Output {
                    $small32($large32(x))
                }
            }

            impl From<$small16> for $large16 {
                fn from(x: $small16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small16> for $large16 {
                type Output = $large16;

                fn cast(x: $small16) -> Self::Output {
                    $large16::from_bits(x.bits >>
                                        ($small16::fbits() -
                                         $large16::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large16> for $small16 {
                type Error = TryFromIntError;

                fn try_from(x: $large16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large16> for $small16 {
                type Output = Result<$small16, Error>;

                fn cast(x: $large16) -> Self::Output {
                    i16(i32(x.bits) <<
                        ($small16::fbits() - $large16::fbits()))
                        .map($small16::from_bits)
                }
            }

            impl From<$small8> for $large16 {
                fn from(x: $small8) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small8> for $large16 {
                type Output = $large16;

                fn cast(x: $small8) -> Self::Output {
                    $large16($small16(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$large16> for $small8 {
                type Error = TryFromIntError;

                fn try_from(x: $large16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$large16> for $small8 {
                type Output = Result<$small8, Error>;

                fn cast(x: $large16) -> Self::Output {
                    $small16(x).map($small8)
                }
            }
        )*

        cast! {
            $(| $medium32 $medium16,)*
            $(- $small32 $small16 $small8,)+
        }
    };

    (- $medium32:ident $medium16:ident $medium8:ident,
     $(- $small32:ident $small16:ident $small8:ident,)*) => {
        // Cast from/to primitives

        impl From<$medium32> for isize {
            fn from(x: $medium32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for isize {
            type Output = Self;

            fn cast(x: $medium32) -> Self::Output {
                isize(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<isize> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: isize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<isize> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: isize) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        impl From<$medium32> for i64 {
            fn from(x: $medium32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for i64 {
            type Output = Self;

            fn cast(x: $medium32) -> Self::Output {
                i64(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i64> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: i64) -> Result<Self, Self::Error> {
                Self::cast(x)
        }
        }

        impl cast::From<i64> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: i64) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        impl From<$medium32> for i32 {
            fn from(x: $medium32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for i32 {
            type Output = Self;

            fn cast(x: $medium32) -> Self::Output {
                i32(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i32> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: i32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i32> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: i32) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        impl From<$medium32> for i16 {
            fn from(x: $medium32) -> Self {
                Self::cast(x)
        }
        }

        impl cast::From<$medium32> for i16 {
            type Output = Self;

            fn cast(x: $medium32) -> Self::Output {
                i16(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i16> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: i16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i16> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: i16) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        impl From<$medium32> for i8 {
            fn from(x: $medium32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for i8 {
            type Output = Self;

            fn cast(x: $medium32) -> Self::Output {
                (x.bits >> $medium32::fbits()) as i8
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i8> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: i8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i8> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: i8) -> Self::Output {
                i32(i64(x) << $medium32::fbits())
                    .map($medium32::from_bits)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium32> for usize {
            type Error = TryFromIntError;

            fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for usize {
            type Output = Result<Self, Error>;

            fn cast(x: $medium32) -> Self::Output {
                usize(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<usize> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: usize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<usize> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: usize) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium32> for u64 {
            type Error = TryFromIntError;

            fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for u64 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium32) -> Self::Output {
                u64(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u64> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: u64) -> Result<Self, Self::Error> {
                Self::cast(x)
        }
        }

        impl cast::From<u64> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: u64) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium32> for u32 {
            type Error = TryFromIntError;

            fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for u32 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium32) -> Self::Output {
                u32(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u32> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: u32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u32> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: u32) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium32> for u16 {
            type Error = TryFromIntError;

            fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for u16 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium32) -> Self::Output {
                u16(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u16> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: u16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u16> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: u16) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium32> for u8 {
            type Error = TryFromIntError;

            fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for u8 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium32) -> Self::Output {
                u8(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u8> for $medium32 {
            type Error = TryFromIntError;

            fn try_from(x: u8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u8> for $medium32 {
            type Output = Result<$medium32, Error>;

            fn cast(x: u8) -> Self::Output {
                i8(x).and_then($medium32)
            }
        }

        impl From<$medium16> for isize {
            fn from(x: $medium16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for isize {
            type Output = Self;

            fn cast(x: $medium16) -> Self::Output {
                isize(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<isize> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: isize) -> Result<Self, Self::Error> {
                Self::cast(x)
        }
        }

        impl cast::From<isize> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: isize) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        impl From<$medium16> for i64 {
            fn from(x: $medium16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for i64 {
            type Output = Self;

            fn cast(x: $medium16) -> Self::Output {
                i64(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i64> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: i64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i64> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: i64) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        impl From<$medium16> for i32 {
            fn from(x: $medium16) -> Self {
                Self::cast(x)
        }
        }

        impl cast::From<$medium16> for i32 {
            type Output = Self;

            fn cast(x: $medium16) -> Self::Output {
                i32(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i32> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: i32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i32> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: i32) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        impl From<$medium16> for i16 {
            fn from(x: $medium16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for i16 {
            type Output = Self;

            fn cast(x: $medium16) -> Self::Output {
                i16(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i16> for $medium16 {
            type Error = <i8 as TryFrom<i16>>::Error;

            fn try_from(x: i16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i16> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: i16) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        impl From<$medium16> for i8 {
            fn from(x: $medium16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for i8 {
            type Output = Self;

            fn cast(x: $medium16) -> Self::Output {
                (x.bits >> $medium16::fbits()) as i8
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i8> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: i8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i8> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: i8) -> Self::Output {
                i16(i32(x) << $medium16::fbits())
                    .map($medium16::from_bits)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium16> for usize {
            type Error = TryFromIntError;

            fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for usize {
            type Output = Result<Self, Error>;

            fn cast(x: $medium16) -> Self::Output {
                usize(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<usize> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: usize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<usize> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: usize) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium16> for u64 {
            type Error = TryFromIntError;

            fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for u64 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium16) -> Self::Output {
                u64(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u64> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: u64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u64> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: u64) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium16> for u32 {
            type Error = TryFromIntError;

            fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for u32 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium16) -> Self::Output {
                u32(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u32> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: u32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u32> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: u32) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium16> for u16 {
            type Error = TryFromIntError;

            fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for u16 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium16) -> Self::Output {
                u16(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u16> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: u16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u16> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: u16) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium16> for u8 {
            type Error = TryFromIntError;

            fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for u8 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium16) -> Self::Output {
                u8(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u8> for $medium16 {
            type Error = TryFromIntError;

            fn try_from(x: u8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u8> for $medium16 {
            type Output = Result<$medium16, Error>;

            fn cast(x: u8) -> Self::Output {
                i8(x).and_then($medium16)
            }
        }

        impl From<$medium8> for isize {
            fn from(x: $medium8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for isize {
            type Output = Self;

            fn cast(x: $medium8) -> Self::Output {
                isize(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<isize> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: isize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<isize> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: isize) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        impl From<$medium8> for i64 {
            fn from(x: $medium8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for i64 {
            type Output = Self;

            fn cast(x: $medium8) -> Self::Output {
                i64(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i64> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: i64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i64> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: i64) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        impl From<$medium8> for i32 {
            fn from(x: $medium8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for i32 {
            type Output = Self;

            fn cast(x: $medium8) -> Self::Output {
                i32(i8(x))
            }

            }
        #[cfg(feature = "try_from")]
        impl TryFrom<i32> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: i32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i32> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: i32) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        impl From<$medium8> for i16 {
            fn from(x: $medium8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for i16 {
            type Output = Self;

            fn cast(x: $medium8) -> Self::Output {
                i16(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i16> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: i16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i16> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: i16) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        impl From<$medium8> for i8 {
            fn from(x: $medium8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for i8 {
            type Output = Self;

            fn cast(x: $medium8) -> Self::Output {
                (x.bits >> $medium8::fbits()) as i8
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<i8> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: i8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<i8> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: i8) -> Self::Output {
                i8(i16(x) << $medium8::fbits())
                    .map($medium8::from_bits)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium8> for usize {
            type Error = TryFromIntError;

            fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for usize {
            type Output = Result<Self, Error>;

            fn cast(x: $medium8) -> Self::Output {
                usize(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<usize> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: usize) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<usize> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: usize) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium8> for u64 {
            type Error = TryFromIntError;

            fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for u64 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium8) -> Self::Output {
                u64(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u64> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: u64) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u64> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: u64) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium8> for u32 {
            type Error = TryFromIntError;

            fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for u32 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium8) -> Self::Output {
                u32(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u32> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: u32) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u32> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: u32) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium8> for u16 {
            type Error = TryFromIntError;

            fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                Self::cast(x)
        }
        }

        impl cast::From<$medium8> for u16 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium8) -> Self::Output {
                u16(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u16> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: u16) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u16> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: u16) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<$medium8> for u8 {
            type Error = TryFromIntError;

            fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for u8 {
            type Output = Result<Self, Error>;

            fn cast(x: $medium8) -> Self::Output {
                u8(i8(x))
            }
        }

        #[cfg(feature = "try_from")]
        impl TryFrom<u8> for $medium8 {
            type Error = TryFromIntError;

            fn try_from(x: u8) -> Result<Self, Self::Error> {
                Self::cast(x)
            }
        }

        impl cast::From<u8> for $medium8 {
            type Output = Result<$medium8, Error>;

            fn cast(x: u8) -> Self::Output {
                i8(x).and_then($medium8)
            }
        }

        // Casts between fixed point numbers

        impl From<$medium16> for $medium32 {
            fn from(x: $medium16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for $medium32 {
            type Output = $medium32;

            fn cast(x: $medium16) -> Self::Output {
                $medium32::from_bits(i32(x.bits) << 16)
            }
        }

        impl From<$medium32> for $medium16 {
            fn from(x: $medium32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for $medium16 {
            type Output = $medium16;

            fn cast(x: $medium32) -> Self::Output {
                $medium16::from_bits((x.bits >> 16) as i16)
            }
        }

        impl From<$medium8> for $medium32 {
            fn from(x: $medium8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for $medium32 {
            type Output = $medium32;

            fn cast(x: $medium8) -> Self::Output {
                $medium32::from_bits(i32(x.bits) << 24)
            }
        }

        impl From<$medium32> for $medium8 {
            fn from(x: $medium32) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium32> for $medium8 {
            type Output = $medium8;

            fn cast(x: $medium32) -> Self::Output {
                $medium8::from_bits((x.bits >> 24) as i8)
            }
        }

        impl From<$medium8> for $medium16 {
            fn from(x: $medium8) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium8> for $medium16 {
            type Output = $medium16;

            fn cast(x: $medium8) -> Self::Output {
                $medium16::from_bits(i16(x.bits) << 8)
            }
        }

        impl From<$medium16> for $medium8 {
            fn from(x: $medium16) -> Self {
                Self::cast(x)
            }
        }

        impl cast::From<$medium16> for $medium8 {
            type Output = $medium8;

            fn cast(x: $medium16) -> Self::Output {
                $medium8::from_bits((x.bits >> 8) as i8)
            }
        }

        $(
            impl From<$small32> for $medium32 {
                fn from(x: $small32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small32> for $medium32 {
                type Output = $medium32;

                fn cast(x: $small32) -> Self::Output {
                    $medium32::from_bits(x.bits >>
                                         ($small32::fbits() -
                                          $medium32::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium32> for $small32 {
                type Error = TryFromIntError;

                fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium32> for $small32 {
                type Output = Result<$small32, Error>;

                fn cast(x: $medium32) -> Self::Output {
                    i32(i64(x.bits) <<
                        ($small32::fbits() - $medium32::fbits()))
                        .map($small32::from_bits)
                }
            }

            impl From<$small16> for $medium32 {
                fn from(x: $small16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small16> for $medium32 {
                type Output = $medium32;

                fn cast(x: $small16) -> Self::Output {
                    $medium32($small32(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium32> for $small16 {
                type Error = TryFromIntError;

                fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium32> for $small16 {
                type Output = Result<$small16, Error>;

                fn cast(x: $medium32) -> Self::Output {
                    $small32(x).map($small16)
                }
            }

            impl From<$small8> for $medium32 {
                fn from(x: $small8) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small8> for $medium32 {
                type Output = $medium32;

                fn cast(x: $small8) -> Self::Output {
                    $medium32($small32(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium32> for $small8 {
                type Error = TryFromIntError;

                fn try_from(x: $medium32) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium32> for $small8 {
                type Output = Result<$small8, Error>;

                fn cast(x: $medium32) -> Self::Output {
                    $small32(x).map($small8)
                }
            }

            impl From<$small32> for $medium16 {
                fn from(x: $small32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small32> for $medium16 {
                type Output = $medium16;

                fn cast(x: $small32) -> Self::Output {
                    $medium16($small16(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium16> for $small32 {
                type Error = TryFromIntError;

                fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium16> for $small32 {
                type Output = Result<$small32, Error>;

                fn cast(x: $medium16) -> Self::Output {
                    $small32($medium32(x))
                }
            }

            impl From<$small16> for $medium16 {
                fn from(x: $small16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small16> for $medium16 {
                type Output = $medium16;

                fn cast(x: $small16) -> Self::Output {
                    $medium16::from_bits(x.bits >>
                                         ($small16::fbits() -
                                          $medium16::fbits()))
                }
            }
            #[cfg(feature = "try_from")]
            impl TryFrom<$medium16> for $small16 {
                type Error = TryFromIntError;

                fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium16> for $small16 {
                type Output = Result<$small16, Error>;

                fn cast(x: $medium16) -> Self::Output {
                    i16(i32(x.bits) <<
                        ($small16::fbits() - $medium16::fbits()))
                        .map($small16::from_bits)
                }
            }

            impl From<$small8> for $medium16 {
                fn from(x: $small8) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small8> for $medium16 {
                type Output = $medium16;

                fn cast(x: $small8) -> Self::Output {
                    $medium16($small16(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium16> for $small8 {
                type Error = TryFromIntError;

                fn try_from(x: $medium16) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium16> for $small8 {
                type Output = Result<$small8, Error>;

                fn cast(x: $medium16) -> Self::Output {
                    $small8($medium8(x))
                }
            }

            impl From<$small32> for $medium8 {
                fn from(x: $small32) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small32> for $medium8 {
                type Output = $medium8;

                fn cast(x: $small32) -> Self::Output {
                    $medium8($small8(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium8> for $small32 {
                type Error = TryFromIntError;

                fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium8> for $small32 {
                type Output = Result<$small32, Error>;

                fn cast(x: $medium8) -> Self::Output {
                    $small32($medium32(x))
                }
            }

            impl From<$small16> for $medium8 {
                fn from(x: $small16) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small16> for $medium8 {
                type Output = $medium8;

                fn cast(x: $small16) -> Self::Output {
                    $medium8($small8(x))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium8> for $small16 {
                type Error = TryFromIntError;

                fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium8> for $small16 {
                type Output = Result<$small16, Error>;

                fn cast(x: $medium8) -> Self::Output {
                    $small16($medium16(x))
                }
            }

            impl From<$small8> for $medium8 {
                fn from(x: $small8) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$small8> for $medium8 {
                type Output = $medium8;

                fn cast(x: $small8) -> Self::Output {
                    $medium8::from_bits(x.bits >>
                                        ($small8::fbits() -
                                         $medium8::fbits()))
                }
            }

            #[cfg(feature = "try_from")]
            impl TryFrom<$medium8> for $small8 {
                type Error = TryFromIntError;

                fn try_from(x: $medium8) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$medium8> for $small8 {
                type Output = Result<$small8, Error>;

                fn cast(x: $medium8) -> Self::Output {
                    i8(i16(x.bits) <<
                       ($small8::fbits() - $medium8::fbits()))
                        .map($small8::from_bits)
                }
            }
        )*

        cast! {
            $(- $small32 $small16 $small8,)*
        }
    };

    () => {};
}

cast! {
    + I31F1,
    + I30F2,
    + I29F3,
    + I28F4,
    + I27F5,
    + I26F6,
    + I25F7,
    + I24F8,
    + I23F9,
    + I22F10,
    + I21F11,
    + I20F12,
    + I19F13,
    + I18F14,
    + I17F15,
    + I16F16,

    | I15F17 I15F1,
    | I14F18 I14F2,
    | I13F19 I13F3,
    | I12F20 I12F4,
    | I11F21 I11F5,
    | I10F22 I10F6,
    | I9F23 I9F7,
    | I8F24 I8F8,

    - I7F25 I7F9 I7F1,
    - I6F26 I6F10 I6F2,
    - I5F27 I5F11 I5F3,
    - I4F28 I4F12 I4F4,
    - I3F29 I3F13 I3F5,
    - I2F30 I2F14 I2F6,
    - I1F31 I1F15 I1F7,
}

macro_rules! to_ixx {
    ($ixx:ident; $($q:ident),+) => {
        $(
            #[cfg(feature = "try_from")]
            impl TryFrom<$q> for $ixx {
                type Error = TryFromIntError;

                fn try_from(x: $q) -> Result<Self, Self::Error> {
                    Self::cast(x)
                }
            }

            impl cast::From<$q> for $ixx {
                type Output = Result<$ixx, Error>;

                fn cast(x: $q) -> Self::Output {
                    $ixx(x.bits >> $q::fbits())
                }
            }
        )+
    }
}

macro_rules! from_uxx {
    ($ixx:ident, $uxx:ident; $($q:ident),+) => {
        $(
            impl From<$uxx> for $q {
                fn from(x: $uxx) -> Self {
                    Self::cast(x)
                }
            }

            impl cast::From<$uxx> for $q {
                type Output = $q;

                fn cast(x: $uxx) -> Self::Output {
                    $q::from_bits($ixx(x) << $q::fbits())
                }
            }
        )+
    }
}

impl From<I16F16> for i16 {
    fn from(x: I16F16) -> Self {
        (x.bits >> I16F16::fbits()) as i16
    }
}

impl cast::From<I16F16> for i16 {
    type Output = Self;

    fn cast(x: I16F16) -> Self::Output {
        (x.bits >> I16F16::fbits()) as i16
    }
}

to_ixx!(i16;
        I31F1,
        I30F2,
        I29F3,
        I28F4,
        I27F5,
        I26F6,
        I25F7,
        I24F8,
        I23F9,
        I22F10,
        I21F11,
        I20F12,
        I19F13,
        I18F14,
        I17F15);

#[cfg(feature = "try_from")]
impl TryFrom<u16> for I16F16 {
    type Error = TryFromIntError;

    fn try_from(x: u16) -> Result<Self, Self::Error> {
        i32::try_from(u32::from(x) << I16F16::fbits()).map(I16F16::from_bits)
    }
}

impl cast::From<u16> for I16F16 {
    type Output = Result<Self, Error>;

    fn cast(x: u16) -> Self::Output {
        i32(u32(x) << I16F16::fbits()).map(I16F16::from_bits)
    }
}

from_uxx!(i32, u16;
          I31F1,
          I30F2,
          I29F3,
          I28F4,
          I27F5,
          I26F6,
          I25F7,
          I24F8,
          I23F9,
          I22F10,
          I21F11,
          I20F12,
          I19F13,
          I18F14,
          I17F15);
impl From<I8F24> for i8 {

    fn from(x: I8F24) -> Self {
        Self::cast(x)
    }
}

impl cast::From<I8F24> for i8 {
    type Output = Self;

    fn cast(x: I8F24) -> Self::Output {
        (x.bits >> I8F24::fbits()) as i8
    }
}

to_ixx!(i8; I15F17, I14F18, I13F19, I12F20, I11F21, I10F22, I9F23);

#[cfg(feature = "try_from")]
impl TryFrom<u8> for I8F24 {
    type Error = Error;

    fn try_from(x: u8) -> Result<Self, Self::Error> {
        Self::cast(x)
    }
}

impl cast::From<u8> for I8F24 {
    type Output = Result<Self, Error>;

    fn cast(x: u8) -> Self::Output {
        i32(u32(x) << I8F24::fbits()).map(I8F24::from_bits)
    }
}

from_uxx!(i32, u8; I15F17, I14F18, I13F19, I12F20, I11F21, I10F22, I9F23);

impl From<I8F8> for i8 {
    fn from(x: I8F8) -> Self {
        Self::cast(x)
    }
}

impl cast::From<I8F8> for i8 {
    type Output = Self;

    fn cast(x: I8F8) -> Self::Output {
        (x.bits >> I8F8::fbits()) as i8
    }
}

to_ixx!(i8; I15F1, I14F2, I13F3, I12F4, I11F5, I10F6, I9F7);

#[cfg(feature = "try_from")]
impl TryFrom<u8> for I8F8 {
    type Error = Error;

    fn try_from(x: u8) -> Result<Self, Self::Error> {
        Self::cast(x)
    }
}

impl cast::From<u8> for I8F8 {
    type Output = Result<Self, Error>;

    fn cast(x: u8) -> Self::Output {
        i16(u16(x) << I8F8::fbits()).map(I8F8::from_bits)
    }
}

from_uxx!(i16, u8; I15F1, I14F2, I13F3, I12F4, I11F5, I10F6, I9F7);

#[cfg(test)]
#[macro_use]
extern crate approx;
mod tests {
    use super::*;
    use cast::f64;
    
    #[test]
    fn atan_works() {
        assert_relative_eq!(
            f64(I16F16(0.0_f64).unwrap().atan()), 
            0.0_f64, 
            epsilon = 1.0e-6);
        assert_relative_eq!(
            f64(I16F16(0.5_f64).unwrap().atan()), 
            0.463648_f64, 
            epsilon = 1.0e-6);
        assert_relative_eq!(
            f64(I16F16(-0.5_f64).unwrap().atan()), 
            -0.463648_f64, 
            epsilon = 1.0e-6);
    
    }
    #[test]
    fn sqrt_works() {
        assert_relative_eq!(
            f64(I16F16(1_f64).unwrap().sqrt().unwrap()), 
            1.0, 
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            f64(I16F16(0_f64).unwrap().sqrt().unwrap()), 
            0.0, 
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            f64(I16F16(0.1_f64).unwrap().sqrt().unwrap()), 
            0.316228, 
            epsilon = 1.0e-4
        );
        assert_relative_eq!(
            f64(I16F16(10.0_f64).unwrap().sqrt().unwrap()), 
            3.16228, 
            epsilon = 1.0e-4
        );
        
    }

    #[test]
    fn rs_works() {
        assert_eq!(f64(I16F16(0_f64).unwrap().rs()), 0.0);
        assert_eq!(f64(I16F16(1_f64).unwrap().rs()), 0.5);
        assert_eq!(f64(I16F16(2_f64).unwrap().rs()), 1.0);
        assert_eq!(f64(I16F16(3_f64).unwrap().rs()), 1.5);
        assert_eq!(f64(I16F16(4_f64).unwrap().rs()), 2.0);
        assert_eq!(f64(I16F16(-1_f64).unwrap().rs()), -0.5);
        assert_eq!(f64(I16F16(-2_f64).unwrap().rs()), -1.0);
        assert_eq!(I16F16::from_bits(1).rs().into_bits(), 1);
        assert_eq!(I16F16::from_bits(2).rs().into_bits(), 1);
        assert_eq!(I16F16::from_bits(3).rs().into_bits(), 2);
        assert_eq!(I16F16::from_bits(4).rs().into_bits(), 2);
    }

    #[test]
    fn log2_works() {
        assert!(I16F16(0_f64).unwrap().log2().is_err());

        assert_eq!(I16F16(1_f64).unwrap().log2().unwrap(), I16F16::zero());

        assert_relative_eq!(
            f64(I16F16(2.0_f64).unwrap().log2().unwrap()), 
            1.0, 
            epsilon = 1.0e-6
        );

        assert_relative_eq!(
            f64(I16F16(4_f64).unwrap().log2().unwrap()), 
            2.0, 
            epsilon = 1.0e-6
        );        

        assert_relative_eq!(
            f64(I16F16(3.33333_f64).unwrap().log2().unwrap()), 
            1.73696, 
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            f64(I16F16(0.11111_f64).unwrap().log2().unwrap()), 
            -3.16994, 
            epsilon = 1.0e-2
        );        

    }

    #[test]
    fn ln_works() {
        assert_eq!(I16F16(1_f64).unwrap().ln().unwrap(), I16F16::zero());

        assert!(I16F16(0_f64).unwrap().ln().is_err());

        assert_relative_eq!(
            f64(I16F16(2.71828_f64).unwrap().ln().unwrap()), 
            1.0, 
            epsilon = 1.0e-4
        );

        assert_relative_eq!(
            f64(I16F16(10_f64).unwrap().ln().unwrap()), 
            2.30259, 
            epsilon = 1.0e-4
        );
        
    }

    #[test]
    fn pow_works() {
        assert_eq!(f64(I16F16(1_f64).unwrap().pow(I16F16(2.0_f64).unwrap())), 1.0);
        assert_relative_eq!(
            f64(I16F16(2_f64).unwrap().pow(I16F16(2.0_f64).unwrap())), 
            4.0,
            epsilon = 1.0e-3
        );
        assert_relative_eq!(
            f64(I16F16(2_f64).unwrap().pow(I16F16(3.0_f64).unwrap())), 
            8.0,
            epsilon = 1.0e-3
        );
        assert_relative_eq!(
            f64(I16F16(2.9_f64).unwrap().pow(I16F16(3.1_f64).unwrap())), 
            27.129,
            epsilon = 1.0e-2
        );
    }

    #[test]
    fn exp_works() {
        assert_eq!(f64(I16F16(0_f64).unwrap().exp()), 1.0);
        assert_relative_eq!(
            f64(I16F16(1_f64).unwrap().exp()), 
            2.718281828459045235,
            epsilon = 1.0e-4
        );

        assert_relative_eq!(
            f64(I16F16(5_f64).unwrap().exp()), 
            148.413159,
            epsilon = 1.0e-3
        );

    }

    #[test]
    fn sin_works() {
        let two_pi          = I8F24(6.283185307179586476925_f64).unwrap();
        let pi              = I8F24(3.141592653589793238462_f64).unwrap();
        let pi_half         = I8F24(1.570796326794896619231_f64).unwrap();
        let pi_quarter      = I8F24(0.785398163397448309615_f64).unwrap();

        assert_relative_eq!(
            f64(I8F24(0_f64).unwrap().sin()), 
            0.0,
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            f64(pi_half.sin()), 
            1.0,
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            f64(pi.sin()), 
            0.0,
            epsilon = 1.0e-5
        );
        
        assert_relative_eq!(
            f64((pi+pi_half).sin()), 
            -1.0,
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            f64(two_pi.sin()), 
            0.0,
            epsilon = 1.0e-5
        );

        assert_relative_eq!(
            f64(pi_quarter.sin()), 
            0.707107,
            epsilon = 1.0e-1
        );

        assert_relative_eq!(
            f64((-pi_half).sin()), 
            -1.0,
            epsilon = 1.0e-1
        );

        assert_relative_eq!(
            f64((-pi_quarter).sin()), 
            -0.707107,
            epsilon = 1.0e-1
        );

        assert_relative_eq!(
            f64((pi + pi_quarter).sin()), 
            -0.707107,
            epsilon = 1.0e-1
        );

        assert_relative_eq!(
            f64(I8F24(2_f64).unwrap().sin()), 
            0.909297,
            epsilon = 1.0e-5
        );
        assert_relative_eq!(
            f64(I8F24(-2_f64).unwrap().sin()), 
            -0.909297,
            epsilon = 1.0e-5
        );

    }

    #[test]
    fn cos_works() {
        let pi_half         = I8F24(1.570796326794896619231_f64).unwrap();
        assert_relative_eq!(
            f64(I8F24(0_f64).unwrap().cos()), 
            1.0,
            epsilon = 1.0e-5
        );    
    }

    #[test]
    fn tan_works() {
        let pi_half         = I8F24(1.570796326794896619231_f64).unwrap();
        assert_relative_eq!(
            f64(I8F24(0_f64).unwrap().tan()), 
            0.0,
            epsilon = 1.0e-5
        );    
        assert_relative_eq!(
            f64(I8F24(1_f64).unwrap().tan()), 
            1.55741,
            epsilon = 1.0e-5
        );    

    }

}
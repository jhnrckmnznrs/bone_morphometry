use anyhow::{bail, Result};

#[derive(Clone, Debug)]
pub struct BinaryVolume {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl BinaryVolume {
    pub fn new(data: Vec<u8>, width: usize, height: usize, depth: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(depth))
            .ok_or_else(|| anyhow::anyhow!("volume dimensions overflow usize"))?;
        if data.len() != expected {
            bail!(
                "volume has {} voxels, but dimensions {}x{}x{} require {}",
                data.len(),
                width,
                height,
                depth,
                expected
            );
        }
        if width < 2 || height < 2 || depth < 2 {
            bail!(
                "all dimensions must be at least 2 for 3-D morphometry; got {}x{}x{}",
                width,
                height,
                depth
            );
        }
        Ok(Self {
            data,
            width,
            height,
            depth,
        })
    }

    #[inline(always)]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.height + y) * self.width + x
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        self.data[self.index(x, y, z)]
    }

    #[inline(always)]
    pub fn get_signed(&self, x: isize, y: isize, z: isize) -> bool {
        if x < 0
            || y < 0
            || z < 0
            || x >= self.width as isize
            || y >= self.height as isize
            || z >= self.depth as isize
        {
            false
        } else {
            self.get(x as usize, y as usize, z as usize) != 0
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn slice_len(&self) -> usize {
        self.width * self.height
    }

    #[inline]
    pub fn same_shape(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.depth == other.depth
    }

    /// Select either the bone or marrow phase, but only inside the ROI.
    pub fn phase_inside(&self, roi: &Self, select_bone: bool) -> Result<Self> {
        if !self.same_shape(roi) {
            bail!(
                "binary image dimensions {}x{}x{} do not match ROI dimensions {}x{}x{}",
                self.width,
                self.height,
                self.depth,
                roi.width,
                roi.height,
                roi.depth
            );
        }

        let data = self
            .data
            .iter()
            .zip(&roi.data)
            .map(|(&bone, &inside)| u8::from(inside != 0 && ((bone != 0) == select_bone)))
            .collect();

        Self::new(data, self.width, self.height, self.depth)
    }

    #[cfg(test)]
    pub fn complement(&self) -> Self {
        let data = self.data.iter().map(|&v| u8::from(v == 0)).collect();
        Self {
            data,
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_inside_distinguishes_marrow_from_outside_roi() {
        let bone = BinaryVolume::new(vec![0, 1, 0, 1, 0, 1, 0, 1], 2, 2, 2).unwrap();
        let roi = BinaryVolume::new(vec![0, 0, 1, 1, 1, 1, 0, 0], 2, 2, 2).unwrap();

        assert_eq!(
            bone.phase_inside(&roi, true).unwrap().data,
            vec![0, 0, 0, 1, 0, 1, 0, 0]
        );
        assert_eq!(
            bone.phase_inside(&roi, false).unwrap().data,
            vec![0, 0, 1, 0, 1, 0, 0, 0]
        );
    }
}

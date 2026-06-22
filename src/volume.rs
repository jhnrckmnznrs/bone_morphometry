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

use std::{error::Error, path::PathBuf};

use plotters::prelude::*;

pub fn plot_cost(history: (Vec<f64>, Vec<f64>), name: &str) -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from("./playground/plotters-doc-data").join(name);
    let root = BitMapBackend::new(path.as_path(), (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Cost", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            0.0..history.0.len() as f64,
            0.0..history.0.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)),
        )?;

    chart
        .configure_mesh()
        .x_desc("Epochs")
        .y_desc("Cost")
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            history.0.iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &RED,
        ))?
        .label("Train")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .draw_series(LineSeries::new(
            history.1.iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &BLUE,
        ))?
        .label("Validation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("Result has been saved to plotters-doc-data/0.png");

    Ok(())
}

pub fn plot_image(
    name: &str,
    img: &Vec<f64>,
    pixel_size: usize,
    text: String,
) -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from("./playground/plotters-doc-data").join(name);
    let img_size = (28 * pixel_size) as u32;
    let mut root = BitMapBackend::new(path.as_path(), (img_size, img_size + 30));

    root.draw_rect(
        (0, 0),
        (img_size as i32, img_size as i32 + 30),
        &WHITE,
        true,
    )?;
    for pixel in 0..784 {
        let x = (pixel % 28 * pixel_size) as i32;
        let y = (pixel / 28 * pixel_size) as i32;

        let pixel_size = pixel_size as i32;
        let intensity = (img[pixel] * 255.0) as u8;

        let color = RGBColor(intensity, intensity, intensity);

        root.draw_rect((x, y), (x + pixel_size, y + pixel_size), &color, true)?;
    }

    root.draw_text(
        &text,
        &TextStyle::from(("sans-serif", 20).into_font()).color(&BLACK),
        (5, img_size as i32 + 5),
    )?;

    root.present()?;
    Ok(())
}

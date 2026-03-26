import { config } from 'dotenv';
import { neon } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-http';
import { migrate } from 'drizzle-orm/neon-http/migrator';

config({ path: '.env.local' });

const sql = neon(process.env.NEON_DATABASE_URL!);
const db = drizzle(sql);

const main = async () => {
  try {
    console.log("Starting migration...");
    await migrate(db, { migrationsFolder: 'migrations' });
    console.log("Migration completed successfully.");
    process.exit(0);
  } catch (error) {
    console.error("Error during migration:", error);
    process.exit(1);
  }
};

main();
